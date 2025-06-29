# src/api/feature_engineering.py

import pandas as pd

def calculate_rfm(df, snapshot_date):
    df = df.copy()
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Value': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    return rfm

def encode_channel(df):
    df = df.copy()
    df['ChannelId_encoded'] = df['ChannelId'].astype('category').cat.codes
    return df
