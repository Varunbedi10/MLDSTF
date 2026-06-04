import pandas as pd
import numpy as np
import joblib
import os


def detect_suspicious(
    input_csv,
    output_csv="suspicious_customers.csv"
):

    # -------------------------------------------------
    # 1. FILE VALIDATION
    # -------------------------------------------------
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"File not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    # -------------------------------------------------
    # 2. REQUIRED COLUMNS
    # -------------------------------------------------
    required_columns = [
        'ClientId',
        'Age',
        'Gender',
        'Occupation',
        'ClientType',
        'SeniorCitizenFlag',
        'RiskCategoryId',
        'total_transactions',
        'total_amount',
        'avg_transaction',
        'max_transaction',
        'min_transaction',
        'recency_days',
        'activity_span',
        'active_days',
        'amount_std',
        'transactions_per_day'
    ]

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # -------------------------------------------------
    # 3. LOAD MODEL FILES
    # -------------------------------------------------
    try:
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans.pkl")
        threshold = joblib.load("threshold.pkl")
        feature_order = joblib.load("feature_order.pkl")
        occupation_encoder = joblib.load(
            "occupation_encoder.pkl"
        )

    except Exception as e:
        raise RuntimeError(
            f"Error loading model files: {e}"
        )

    # -------------------------------------------------
    # 4. BASIC CLEANING
    # -------------------------------------------------
    df = df.copy()

    df['Age'] = df['Age'].fillna(
        df['Age'].median()
    )

    df['Gender'] = df['Gender'].fillna(0)

    df['Occupation'] = df['Occupation'].fillna(
        "UNKNOWN"
    )

    df['SeniorCitizenFlag'] = (
        df['SeniorCitizenFlag']
        .fillna(
            df['SeniorCitizenFlag']
            .mode()[0]
        )
    )

    df['RiskCategoryId'] = (
        df['RiskCategoryId']
        .fillna(
            df['RiskCategoryId']
            .mode()[0]
        )
    )

    df['amount_std'] = (
        df['amount_std']
        .fillna(0)
    )

    # -------------------------------------------------
    # 5. AGE VALIDATION
    # -------------------------------------------------
    df.loc[
        (df['Age'] < 18) |
        (df['Age'] > 100),
        'Age'
    ] = df['Age'].median()

    # -------------------------------------------------
    # 6. OCCUPATION ENCODING
    # -------------------------------------------------
    try:
        df['Occupation'] = (
            occupation_encoder.transform(
                df['Occupation']
            )
        )

    except Exception:
        raise ValueError(
            "Occupation contains unseen values."
        )

    # -------------------------------------------------
    # 7. LOG TRANSFORMATION
    # -------------------------------------------------
    log_cols = [
        'total_amount',
        'avg_transaction',
        'max_transaction',
        'amount_std',
        'transactions_per_day'
    ]

    for col in log_cols:

        df[col] = np.log1p(
            df[col].clip(lower=0)
        )

    # -------------------------------------------------
    # 8. FEATURE ALIGNMENT
    # -------------------------------------------------
    features = df[
        feature_order
    ].copy()

    # -------------------------------------------------
    # 9. SCALING
    # -------------------------------------------------
    scaled = scaler.transform(
        features
    )

    # -------------------------------------------------
    # 10. DISTANCE CALCULATION
    # -------------------------------------------------
    distances = kmeans.transform(
        scaled
    )

    df['distance'] = np.min(
        distances,
        axis=1
    )

    # -------------------------------------------------
    # 11. ANOMALY DETECTION
    # -------------------------------------------------
    df['anomaly'] = (
        df['distance'] > threshold
    )

    # -------------------------------------------------
    # 12. RULE BASED FLAGS
    # -------------------------------------------------
    high_amount_threshold = (
        df['total_amount']
        .quantile(0.95)
    )

    high_variance_threshold = (
        df['amount_std']
        .quantile(0.95)
    )

    high_frequency_threshold = (
        df['transactions_per_day']
        .quantile(0.95)
    )

    df['high_amount_flag'] = (
        df['total_amount']
        > high_amount_threshold
    )

    df['high_variance_flag'] = (
        df['amount_std']
        > high_variance_threshold
    )

    df['high_frequency_flag'] = (
        df['transactions_per_day']
        > high_frequency_threshold
    )

    # -------------------------------------------------
    # 13. RISK SCORE
    # -------------------------------------------------
    df['risk_score'] = (
        df['anomaly'].astype(int) * 2 +
        df['high_amount_flag'].astype(int) +
        df['high_variance_flag'].astype(int) +
        df['high_frequency_flag'].astype(int)
    )

    # -------------------------------------------------
    # 14. RISK LEVEL
    # -------------------------------------------------
    df['risk_level'] = 'LOW'

    df.loc[
        df['risk_score'] >= 4,
    'risk_level'
    ] = 'MEDIUM'

    df.loc[
    df['risk_score'] >= 4,
    'risk_level'
    ] = 'HIGH'

    df.loc[
    df['risk_score'] >= 5,
    'risk_level'
    ] = 'CRITICAL'

    # -------------------------------------------------
    # 15. FRAUD EXPLANATION
    # -------------------------------------------------
    def generate_reason(row):

        reasons = []

        if row['anomaly']:
            reasons.append(
                "Anomalous behavior"
            )

        if row['high_amount_flag']:
            reasons.append(
                "High transaction amount"
            )

        if row['high_variance_flag']:
            reasons.append(
                "High transaction variance"
            )

        if row['high_frequency_flag']:
            reasons.append(
                "High transaction frequency"
            )

        return ", ".join(reasons)


    df['reason'] = df.apply(
        generate_reason,
        axis=1
    )

    # -------------------------------------------------
    # 16. FINAL SUSPICIOUS LOGIC
    # -------------------------------------------------
    df['suspicious'] = (
        (df['anomaly'] == True)
        &
        (df['risk_score'] >= 4)
    )

    # -------------------------------------------------
    # 17. DISTRIBUTION REPORT
    # -------------------------------------------------
    print("\n📊 Suspicious Distribution")
    print(
        df['suspicious']
        .value_counts()
    )

    print("\n📊 Risk Score Distribution")
    print(
        df['risk_score']
        .value_counts()
    )

    print("\n📊 Distance Summary")
    print(
        df['distance']
        .describe()
    )

    # -------------------------------------------------
    # 18. OUTPUT
    # -------------------------------------------------
    output_cols = [
        'ClientId',
        'distance',
        'risk_score',
        'risk_level',
        'reason',
        'anomaly',
        'high_amount_flag',
        'high_variance_flag',
        'high_frequency_flag',
        'total_amount',
        'avg_transaction',
        'total_transactions',
        'suspicious'
    ]

    output = df[
        df['suspicious']
    ][output_cols]

    output.to_csv(
        output_csv,
        index=False
    )

    print(
        f"\n✅ Suspicious customers saved to: {output_csv}"
    )

    print(
        f"🚨 Total suspicious users: {len(output)}"
    )

    return output


if __name__ == "__main__":

    file_path = input(
        "Enter CSV file path: "
    ).strip()

    try:
        detect_suspicious(
            file_path
        )

    except Exception as e:

        print(
            f"❌ Error: {e}"
        )