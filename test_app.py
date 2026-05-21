import pandas as pd


def test_csv_columns():

    data = {
        'Date': ['2025-01-01', '2025-01-02'],
        'Sales': [100, 200]
    }

    df = pd.DataFrame(data)

    assert 'Date' in df.columns
    assert 'Sales' in df.columns



def test_sales_values():

    sales = [100, 200, 300]

    assert max(sales) == 300
    assert min(sales) == 100