import pandas as pd
import numpy as np
import joblib
import os


def detect_suspicious(input_csv, output_csv="suspicious_customers.csv"):

    # -----------------------------
    # 1. VALIDATION
    # -----------------------------
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = ['ClientId', 'TransactionAmount', 'TransactionDate']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -----------------------------
    # 2. DATA PREPARATION
    # -----------------------------
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    df = df.dropna(subset=['TransactionDate', 'TransactionAmount'])

    # -----------------------------
    # 3. FEATURE ENGINEERING
    # -----------------------------
    agg = df.groupby('ClientId').agg({
        'TransactionAmount': ['count', 'sum', 'mean', 'max', 'min', 'std'],
        'TransactionDate': ['min', 'max', 'nunique']
    })

    agg.columns = [
        'total_transactions',
        'total_amount',
        'avg_transaction',
        'max_transaction',
        'min_transaction',
        'amount_std',
        'first_txn',
        'last_txn',
        'active_days'
    ]

    agg = agg.reset_index()

    # Handle missing std
    agg['amount_std'] = agg['amount_std'].fillna(0)

    # Time-based features
    today = pd.Timestamp.today()
    agg['recency_days'] = (today - agg['last_txn']).dt.days
    agg['activity_span'] = (agg['last_txn'] - agg['first_txn']).dt.days

    # -----------------------------
    # ADD REQUIRED FEATURES
    # -----------------------------
    agg['transactions_per_day'] = agg['total_transactions'] / agg['active_days'].replace(0, 1)
    agg['amount_var'] = agg['amount_std'] ** 2
    agg['total_accounts'] = 1  # placeholder

    # -----------------------------
    # 4. LOAD MODEL FILES
    # -----------------------------
    try:
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans.pkl")
        threshold = joblib.load("threshold.pkl")
        feature_order = joblib.load("feature_order.pkl")
    except Exception as e:
        raise RuntimeError(f"Error loading model files: {e}")

    # -----------------------------
    # 5. FEATURE ALIGNMENT
    # -----------------------------
    missing_features = [f for f in feature_order if f not in agg.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    features = agg[feature_order]

    # -----------------------------
    # 6. SCALING + DISTANCE
    # -----------------------------
    scaled = scaler.transform(features)

    distances = kmeans.transform(scaled)
    agg['distance'] = np.min(distances, axis=1)

    # -----------------------------
    # 7. ANOMALY DETECTION (CORE ML)
    # -----------------------------
    agg['anomaly'] = agg['distance'] > threshold

    # -----------------------------
    # 8. RULE-BASED FLAGS
    # -----------------------------
    high_amount_threshold = agg['avg_transaction'].quantile(0.98)
    high_variance_threshold = agg['amount_std'].quantile(0.98)

    agg['high_amount_flag'] = agg['avg_transaction'] > high_amount_threshold
    agg['high_variance_flag'] = agg['amount_std'] > high_variance_threshold

    # -----------------------------
    # 9. RISK SCORING (FINAL)
    # -----------------------------
    agg['risk_score'] = (
        agg['anomaly'].astype(int) * 2 +
        agg['high_amount_flag'].astype(int) +
        agg['high_variance_flag'].astype(int)
    )

    # -----------------------------
    # FINAL SUSPICIOUS LOGIC
    # -----------------------------
    amount_threshold = max(
        agg['total_amount'].quantile(0.92),
        2_000_000   
    )

    avg_txn_threshold = max(
        agg['avg_transaction'].quantile(0.92),
        150_000   
    )

    agg['suspicious'] = (
        (agg['anomaly'] == True) &
        (agg['risk_score'] >= 3) &
        (
            (agg['total_amount'] > amount_threshold) &
            (agg['avg_transaction'] > avg_txn_threshold)
        )
    )

    # -----------------------------
    # DISTRIBUTION CHECKS
    # -----------------------------
    print("\n📊 Suspicious Distribution:")
    print(agg['suspicious'].value_counts())

    print("\n📊 Risk Score Distribution:")
    print(agg['risk_score'].value_counts())

    print("\n📊 Distance Summary:")
    print(agg['distance'].describe())

    # -----------------------------
    # 10. OUTPUT
    # -----------------------------
    output = agg[agg['suspicious'] == True]

    output.to_csv(output_csv, index=False)

    print(f"\n✅ Suspicious customers saved to: {output_csv}")
    print(f"🚨 Total suspicious users: {len(output)}")

    return output


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    file_path = input("Enter CSV file path: ").strip()

    try:
        detect_suspicious(file_path)
    except Exception as e:
        print(f"❌ Error: {e}")