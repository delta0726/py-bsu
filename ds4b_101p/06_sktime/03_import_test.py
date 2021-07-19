# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 6 (Sktime): ARIMA Automation
# File        : 03_import_test.py
# Update Date : 2021/7/3
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************

# Imports

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast

from sktime.forecasting.arima import AutoARIMA
from tqdm import tqdm


# データロード
df = collect_data()

# データ集計
# --- 単一系列
bike_sales_m_df = df\
    .summarize_by_time(date_column='order_date',
                       value_column='total_price',
                       rule='M',
                       kind='period')

# データ集計
# --- 複数系列
bike_sales_cat2_m_df = df\
    .summarize_by_time(date_column='order_date',
                       value_column='total_price',
                       groups=['category_2'],
                       rule='M',
                       kind='period')

# 動作テスト
# --- 単一系列
bike_sales_m_df.arima_forecast(h=12, sp=1)

# 動作テスト
# --- 複数系列
bike_sales_cat2_m_df\
    .arima_forecast(h=12, sp=1)\
    .groupby('category_2')\
    .plot(x='order_date')

plt.show()