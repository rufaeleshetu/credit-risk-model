import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "data/processed/xente_transactions_processed.csv"
OUTPUT_PATH = "data/processed/xente_transactions_engineered.csv"

def encode_categorical_columns(df, columns):
    """Encodes categorical columns using Label Encoding."""
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def calculate_aggregate_features(df):
    """Generates aggregate features from transactional data."""
    if 'CustomerId' in df.columns:
        agg = df.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'AccountId': pd.Series.nunique,
            'ProductId': pd.Series.nunique
        }).rename(columns={
            'TransactionId': 'TransactionCount',
            'AccountId': 'UniqueAccounts',
            'ProductId': 'UniqueProducts'
        }).reset_index()
        df = df.merge(agg, on='CustomerId', how='left')
    return df

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ File not found at: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)

    # Time features are skipped due to missing datetime column

    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory']
    df = encode_categorical_columns(df, categorical_cols)

    df = calculate_aggregate_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Engineered features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
