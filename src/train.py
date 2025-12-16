import pandas as pd
# import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


RAW_PATH = Path("../data/data.csv")
PROCESSED_PATH = Path("../data/processed/processed_data.csv")
FINAL_PATH = Path("../data/processed/final_training_data.csv")


def create_rfm_target():
    df = pd.read_csv(RAW_PATH, parse_dates=["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (
            snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    ).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    high_risk_cluster = rfm.groupby(
        "cluster")[["frequency", "monetary"]].mean().idxmin()[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    features = pd.read_csv(PROCESSED_PATH)
    final_df = features.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    final_df.to_csv(FINAL_PATH, index=False)
    print("Proxy target variable created and merged.")


if __name__ == "__main__":
    create_rfm_target()
