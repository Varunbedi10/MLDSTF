import streamlit as st
import pandas as pd
import os
import plotly.express as px
from app import detect_suspicious


 
st.set_page_config(
    page_title="TFL",
    layout="wide"
)


st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["Dashboard", "Full Data Analysis", "Suspicious Customers"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI Fraud Detection System")



st.title("🚨 SUSPICIOUS CUSTOMER DETECTION DASHBOARD")
st.markdown("### TRUST FINTECH Ltd.")
st.divider()



uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to begin analysis.")
    st.stop()

temp_file = "temp_uploaded.csv"

with open(temp_file, "wb") as f:
    f.write(uploaded_file.getbuffer())

df = pd.read_csv(temp_file)



with st.spinner("Running Fraud Detection Model..."):
    result = detect_suspicious(temp_file)

total_customers = len(df)
suspicious_count = len(result)

fraud_rate = 0
if total_customers > 0:
    fraud_rate = round((suspicious_count / total_customers) * 100, 2)


# =====================================
# PAGE 1 - DASHBOARD
# =====================================
if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Suspicious Customers", suspicious_count)
    col3.metric("Fraud Rate (%)", fraud_rate)

    st.divider()

    col4, col5 = st.columns(2)

    # Risk Level Distribution
    if suspicious_count > 0:
        fig1 = px.histogram(
            result,
            x="risk_level",
            color="risk_level",
            title="Risk Level Distribution"
        )
        col4.plotly_chart(fig1, use_container_width=True)

    # Total Amount Distribution
    if "total_amount" in df.columns:
        fig2 = px.histogram(
            df,
            x="total_amount",
            nbins=50,
            title="Total Transaction Amount Distribution"
        )
        col5.plotly_chart(fig2, use_container_width=True)




elif page == "Full Data Analysis":
    st.subheader("📊 Dataset Overview")
st.dataframe(df, use_container_width=True)

st.divider()

st.subheader("📈 Quick Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))

if "total_amount" in df.columns:

    avg_amount = round(df["total_amount"].mean(), 2)
    max_amount = df["total_amount"].max()
    min_amount = df["total_amount"].min()

    col2.metric(
        "Average Amount",
        f"₹{avg_amount:,.2f}"
    )

    col3.metric(
        "Highest Amount",
        f"₹{max_amount:,.2f}"
    )

    col4.metric(
        "Lowest Amount",
        f"₹{min_amount:,.2f}"
    )

st.divider()

if "total_amount" in df.columns and "client_id" in df.columns:

    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button("🔍 Show Highest Transaction Client"):

            highest_customer = df.loc[
                df["total_amount"].idxmax()
            ]

            st.success(
                f"Highest Transaction Client ID: {highest_customer['client_id']}"
            )

    with btn2:
        if st.button("🔍 Show Lowest Transaction Client"):

            lowest_customer = df.loc[
                df["total_amount"].idxmin()
            ]

            st.success(
                f"Lowest Transaction Client ID: {lowest_customer['client_id']}"
            )

st.divider()

col1, col2 = st.columns(2)

if "Age" in df.columns:
    fig_age = px.histogram(
        df,
        x="Age",
        nbins=30,
        title="Age Distribution"
    )
    col1.plotly_chart(fig_age, use_container_width=True)

if "transactions_per_day" in df.columns:
    fig_freq = px.histogram(
        df,
        x="transactions_per_day",
        nbins=40,
        title="Transactions Per Day Distribution"
    )
    col2.plotly_chart(fig_freq, use_container_width=True)




# =====================================
# PAGE 3 - SUSPICIOUS CUSTOMERS
# =====================================
elif page == "Suspicious Customers":

    if suspicious_count == 0:
        st.success("✅ No suspicious customers detected.")
    else:
        st.subheader("🚨 Suspicious Customer List")

        # Risk Level Filter
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=result["risk_level"].unique(),
            default=result["risk_level"].unique()
        )

        filtered = result[result["risk_level"].isin(risk_filter)]

        st.dataframe(filtered, use_container_width=True)

        # Download button
        csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_suspicious_customers.csv",
            mime="text/csv"
        )

        st.divider()

        col1, col2 = st.columns(2)

        # Risk Score Distribution
        fig_score = px.histogram(
            filtered,
            x="risk_score",
            title="Risk Score Distribution"
        )
        col1.plotly_chart(fig_score, use_container_width=True)

        # Transactions vs Amount
        if "total_amount" in filtered.columns:
            fig_amt = px.scatter(
                filtered,
                x="total_transactions",
                y="total_amount",
                color="risk_level",
                title="Transactions vs Total Amount"
            )
            col2.plotly_chart(fig_amt, use_container_width=True)


# =====================================
# FOOTER
# =====================================
st.divider()
st.caption("Trust Fintech Ltd. - Suspicious Customer Detection System")