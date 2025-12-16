#
from src.feature_engineering import create_aggregate_features, create_time_features, prepare_base_table
import sys
import pandas as pd
import pytest
sys.path.append('..')
# Sample data for testing (small subset of your original dataset)


@pytest.fixture
def sample_data():
    data = {
        'CustomerId': [1, 2, 1, 2, 3],
        'TransactionId': [101, 102, 103, 104, 105],
        'Amount': [100, 200, 150, 250, 300],
        'TransactionStartTime': [
            '2025-12-01 08:00:00', '2025-12-02 09:00:00', '2025-12-01 10:00:00',
            '2025-12-03 11:00:00', '2025-12-02 12:00:00'
        ],
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# Test for create_aggregate_features


def test_create_aggregate_features(sample_data):
    agg_features = create_aggregate_features(sample_data)
    assert 'CustomerId' in agg_features.columns
    assert 'total_amount' in agg_features.columns
    assert 'avg_amount' in agg_features.columns
    # There are 3 unique customers in the sample data
    assert agg_features.shape[0] == 3

# Test for create_time_features


def test_create_time_features(sample_data):
    df_time = create_time_features(sample_data)
    assert 'transaction_hour' in df_time.columns
    assert 'transaction_day' in df_time.columns
    assert 'transaction_month' in df_time.columns
    assert 'transaction_year' in df_time.columns
    assert df_time.shape[0] == sample_data.shape[0]

# Test for prepare_base_table


def test_prepare_base_table(sample_data):
    base_table, cat_cols, num_cols = prepare_base_table(sample_data)

    # Test if the function returns the expected output
    assert 'CustomerId' in base_table.columns
    assert 'transaction_hour' in base_table.columns
    assert 'transaction_day' in base_table.columns
    assert 'total_amount' in base_table.columns
    assert isinstance(cat_cols, list)
    assert isinstance(num_cols, list)
