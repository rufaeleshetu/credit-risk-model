import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.feature_engineering import calculate_rfm, encode_channel

def test_calculate_rfm():
    # Sample data
    df = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B'],
        'TransactionStartTime': ['2025-06-01', '2025-06-10', '2025-05-01'],
        'Value': [100, 200, 300]
    })
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # RFM computation
    snapshot_date = datetime.strptime('2025-07-01', '%Y-%m-%d')
    rfm = calculate_rfm(df, snapshot_date)

    # Assertions
    assert set(['Recency', 'Frequency', 'Monetary']).issubset(rfm.columns)
    assert rfm.shape[0] == 2  # 2 customers

def test_encode_channel():
    df = pd.DataFrame({'ChannelId': ['web', 'IOS', 'Android']})
    encoded = encode_channel(df)
    
    # Basic check for encoded output
    assert 'ChannelId_encoded' in encoded.columns
    assert encoded['ChannelId_encoded'].notnull().all()
