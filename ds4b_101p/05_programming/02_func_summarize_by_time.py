# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 5 (Programming): Functions
# File        : 02_func_summarize_by_time.py
# Update Date : 2021/7/2
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# Imports

import pandas as pd
import numpy as np
from pandas.core import groupby

from my_pandas_extensions.database import collect_data

# データロード
df = collect_data()


rule = "Q"

# WHAT WE WANT TO STREAMLINE
# --- category2: Grouping
# --- order_date: Date
# --- total_price: Value
df[['category_2', 'order_date', 'total_price']] \
    .set_index('order_date') \
    .groupby('category_2') \
    .resample(rule, kind='period') \
    .agg(np.sum) \
    .unstack("category_2") \
    .reset_index() \
    .assign(order_date=lambda x: x['order_date'].dt.to_period()) \
    .set_index("order_date")


# BUILDING SUMMARIZE BY TIME

def summarize_by_time(data, date_column, value_column, groups=None, rule="D",
                      agg_func=np.sum, kind="timestamp", wide_format=True,
                      fillna=0, *args, **kwargs):
    """
    Used to detect outliers using the 1.5 IQR (Inner Quartile Range) Method.

    Args:
        x (Pandas Series):
            A numeric pandas series.

        iqr_multiplier (int, float, optional):
            A multiplier used to modify the IQR sensitivity.
            Must be positive. Lower values will add more outliers.
            Larger values will add fewer outliers. Defaults to 1.5.

        how (str, optional):
            One of "both", "upper" or "lower". Defaults to "both".
            - "both": flags both upper and lower outliers.
            - "lower": flags lower outliers only.
            - "upper": flags upper outliers only.

    Returns:
        [Pandas Series]: A Boolean Series that flags outliers as True/False.
    """

    # Checks
    if type(data) is not pd.DataFrame:
        raise TypeError("'data' is not pandas DataFrame")

    # --- Seriesの場合はデータフレーム変換
    if type(value_column) is not list:
        value_column = [value_column]

    # Body
    # Handle date column
    data = data.set_index(date_column)

    # Handle Group
    if groups is not None:
        data = data.groupby(groups)

    # Handle Resample
    data = data.resample(rule=rule, kind=kind)

    # Handle aggregation
    function_list = [agg_func] * len(value_column)
    agg_dict = dict(zip(value_column, function_list))
    data = data[value_column].agg(func=agg_dict, *args, **kwargs)

    # Handle Pivot Wider
    if wide_format:
        if groups is not None:
            data = data.unstack(groups)
            if kind == "period":
                data.index = data.index.to_period()

    data = data.fillna(fillna)

    return data


# 動作確認 ------------------------------------------------------------------

# データ格納
data = df

# 関数実行
summarize_by_time(data,
                  date_column="order_date",
                  value_column=["total_price", "quantity"],
                  groups=["category_1", "category_2"],
                  rule="D",
                  kind="period",
                  agg_func=[np.sum, np.mean],
                  wide_format=True,
                  fillna=np.nan)



# ADDING TO OUR TIME SERIES MODULE
{
    "total_price": np.sum,
     "quantity": np.sum
}

value_column = ["total_price", "quantity"]
agg_func = np.sum

[agg_func] * len(value_column)

function_list = [agg_func] * len(value_column)

dict(zip(value_column, function_list))


# エラー出力チェック
summarize_by_time("abc", "order_date", "total_price")