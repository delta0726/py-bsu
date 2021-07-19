# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 8 (SQL Database Update): Forecast Automation
# File        : 02_forecast_db_automation.py
# Update Date : 2021/7/15
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************



# 0 準備 --------------------------------------------------------------------

import pandas as pd
import numpy as np

from my_pandas_extensions.database import (
    collect_data,
    convert_to_datetime, 
    write_forecast_to_database, 
    read_forecast_from_database, 
    prep_forecast_data_for_update
)

from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast, plot_forecast


# データロード
df = collect_data()


# 1 SUMMARIZE AND FORECAST ------------------------------------------------

# 予測データの作成
forecast_1_df = df\
    .summarize_by_time(date_column='order_date', 
                       value_column='total_price', 
                       rule='M', 
                       kind='period')\
    .arima_forecast(h=12, sp=12)\
    .assign(id='Total Revenue')

# データ加工
forecast_1_df = forecast_1_df\
    .prep_forecast_data_for_update(
        id_column='id', 
        date_column='order_date'
    )

# プロット作成
forecast_1_df.plot_forecast(id_column='id', date_column='date')



# 1.2 Revenue by Category 1 ----------------------------------------------

# 予測データの作成
forecast_2_df = df\
    .summarize_by_time(date_column='order_date', 
                       value_column='total_price',  
                       groups='category_1', 
                       rule='M', 
                       kind='period')\
    .arima_forecast(h=12, sp=12)

# データ加工
forecast_2_df = forecast_2_df\
    .prep_forecast_data_for_update(date_column='order_date', 
                                   id_column='category_1')

# プロット作成
pd.concat([forecast_1_df, 
           forecast_2_df], axis=0)\
    .plot_forecast(id_column='id', 
                   date_column='date')


# 1.3 Revenue by Category 2 --------------------------------------------

# 予測データの作成
forecast_3_df = df\
    .summarize_by_time(date_column='order_date', 
                       value_column='total_price', 
                       groups='category_2', 
                       rule='M', 
                       kind='period')\
    .arima_forecast(h=12, sp=12)

# データ加工
forecast_3_df = forecast_3_df\
    .prep_forecast_data_for_update(date_column='order_date', 
                                   id_column='category_2')

# プロット作成
pd.concat([forecast_1_df, 
           forecast_2_df, 
           forecast_3_df], axis=0)\
    .plot_forecast(id_column='id', date_column='date', facet_ncol=3)


# 1.4 Revenue by Customer --------------------------------------------

# 予測データの作成
forecast_4_df = df\
    .summarize_by_time(date_column='order_date', 
                       value_column='total_price', 
                       groups='bikeshop_name', 
                       rule='Q', 
                       kind='period')\
    .arima_forecast(h=4, sp=4)


# データ加工
forecast_4_df = forecast_4_df\
    .prep_forecast_data_for_update(date_column='order_date', 
                                   id_column='bikeshop_name')

# プロット作成
pd.concat([forecast_1_df, 
           forecast_2_df, 
           forecast_3_df, 
           forecast_4_df], axis=0)\
    .plot_forecast(id_column='id', date_column='date', facet_ncol=3)


# 2.0 UPDATE DATABASE --------------------------------------------

# データ結合
all_forecast_df = \
    pd.concat([forecast_1_df, 
               forecast_2_df, 
               forecast_3_df, 
               forecast_4_df], axis=0)

# データ保存＆確認
all_forecast_df.to_pickle("08_sql_database_update/all_forecast_df.pkl")
all_forecast_df = pd.read_pickle("08_sql_database_update/all_forecast_df.pkl")

# 2.1 Write to Database ----

all_forecast_df\
    .write_forecast_to_database(id_column='id', 
                                date_column='date', 
                                if_exists='replace')


def convert_to_datetime(data, date_column):

    df_prepped = data
    
    if df_prepped[date_column].dtype is not 'datetime64[ns]':
        try:
            df_prepped[date_column] = df_prepped[date_column].dt.to_timestamp()
        except:
            try: 
                df_prepped[date_column] = pd.to_datetime(df_prepped[date_column])
            except:
                raise Exception("Could not auto-convert `date_column` to datetime64.")
    
    return data

convert_to_datetime(all_forecast_df, date_column='date')


# 2.2 Read from Database ------------------------------------------------

# データロード
all_forecast_from_db_df = read_forecast_from_database()

# データ確認
all_forecast_from_db_df.info
all_forecast_df.info
