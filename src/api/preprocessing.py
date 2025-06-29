import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=['TransactionId', 'BatchId', 'SubscriptionId'], errors='ignore')

    # Handle datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    # Aggregate per customer
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum', 'mean'],
        'TransactionHour': 'mean',
        'TransactionMonth': 'nunique',
        'FraudResult': 'max'
    }).reset_index()

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    return agg_df


def build_preprocessing_pipeline(X: pd.DataFrame, y: pd.Series = None):
    # WOE transformation
    woe = WOE()
    X_woe = woe.fit_transform(X, y)

    numeric_features = [col for col in X_woe.columns if col not in ['CustomerId']]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    return preprocessor, X_woe, woe
