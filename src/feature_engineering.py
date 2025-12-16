from os import name
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
import joblib
from pathlib import Path


RAW_PATH = Path("../data/data.csv")
PROCESSED_PATH = Path("../data/processed/processed_data.csv")
PIPELINE_PATH = Path("../data/processed/feature_pipeline.pkl")


def load_data():
    return pd.read_csv(RAW_PATH, parse_dates=["TransactionStartTime"])


def create_aggregate_features(df):
    agg = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_amount=("Amount", "std"),
    ).reset_index()

    agg["std_amount"] = agg["std_amount"].fillna(0)
    return agg


def create_time_features(df):
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df


def prepare_base_table(df):
    categorical_cols = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    df = create_time_features(df)

    df_base = df.groupby("CustomerId").agg({
        **{col: "first" for col in categorical_cols},
        "Value": "mean",
        "transaction_hour": "mean",
        "transaction_day": "mean",
        "transaction_month": "mean",
        "transaction_year": "first",
    }).reset_index()

    numerical_cols = [
        "Value",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    return df_base, categorical_cols, numerical_cols

    """df = create_time_features(df)

    categorical_cols = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    numerical_cols = [
        "Value",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    df_base = df[["CustomerId"] + categorical_cols + numerical_cols]

    # df_base = df_base.groupby("CustomerId").first().reset_index()
    df_base = df_base.groupby("CustomerId").agg({
        "CurrencyCode": "first",
        "CountryCode": "first",
        "ProviderId": "first",
        "ProductCategory": "first",
        "ChannelId": "first",
        "PricingStrategy": "first",
        "Value": "mean",
        "transaction_hour": "mean",
        "transaction_day": "mean",
        "transaction_month": "mean",
        "transaction_year": "first",
    }).reset_index()

    return df_base, categorical_cols, numerical_cols
    """


def build_pipeline(cat_cols, num_cols):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    return preprocessor


def run_feature_engineering():
    df = load_data()

    agg_features = create_aggregate_features(df)
    base_table, cat_cols, num_cols = prepare_base_table(df)

    df_merged = base_table.merge(agg_features, on="CustomerId", how="left")

    full_num_cols = num_cols + [
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
    ]

    pipeline = build_pipeline(cat_cols, full_num_cols)

    X_transformed = pipeline.fit_transform(
        df_merged.drop(columns=["CustomerId"]))

    feature_names = (
        full_num_cols +
        list(pipeline.named_transformers_["cat"]
             .named_steps["encoder"]
             .get_feature_names_out(cat_cols))
    )

    processed_df = pd.DataFrame(X_transformed, columns=feature_names)
    processed_df["CustomerId"] = df_merged["CustomerId"].values

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)

    joblib.dump(pipeline, PIPELINE_PATH)

    print("Feature engineering completed and saved.")


if name == "main":
    run_feature_engineering()
