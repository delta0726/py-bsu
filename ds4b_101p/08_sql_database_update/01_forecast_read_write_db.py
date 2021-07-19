# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 8 (SQL Database Update): Forecast Write and Read Functions
# File        : 01_forecast_read_write_db.py
# Update Date : 2021/7/2
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備


# 0 準備 ---------------------------------------------------------------------
# IMPORTS ----

import sqlalchemy as sql
from sqlalchemy.sql.schema import MetaData
from sqlalchemy.types import String, Numeric

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast, plot_forecast


# データロード
df = collect_data()


# WORKFLOW ----
# - Until Module 07: Visualization

# 予測データの作成
arima_forecast_df = df\
    .summarize_by_time(date_column='order_date', value_column='total_price', 
                       groups='category_2', rule='M', agg_func=np.sum, 
                       kind='period', wide_format=True, fillna=0)\
    .arima_forecast(h=12, sp=1, suppress_warning=True, alpha=0.05)


# プロット作成
arima_forecast_df\
    .plot_forecast(id_column='category_2', 
                   date_column='order_date', 
                   facet_ncol=3)


arima_forecast_df\
    .rename({'category_2': 'id', 
             'order_date': 'date'}, axis=1)


# DATABASE UPDATE FUNCTIONS ----


# 1.0 PREPARATION FUNCTIONS ----------------------------------------


# 関数定義
def prep_forecast_data_for_update(data, id_column, date_column):
    
    # Format Column
    df = data.rename({id_column: 'id', 
                      date_column: 'date'}, axis=1)

    # Validate correct column
    required_col_name = ['id', 'date', 'value', 'prediction', 'ci_lo', 'ci_hi']
    
    if not all(pd.Series(required_col_name).isin(df.columns)): 
        col_text = ", ".join(required_col_name)
        raise Exception (f"Columns must contain: {col_text}")
    
    
    # Output
    return(df)


# 動作確認
# --- エラー発生用： data=arima_forecast_df.drop("ci_lo", axis=1)
prep_forecast_data_for_update(arima_forecast_df, 
                              id_column='category_2', 
                              date_column='order_date')


# 2.0 WRITE TO DATABASE ----------------------------------------

# 関数定義
def write_forecast_to_database(data, id_column, date_column, 
                               conn_string='sqlite:///00_database/bike_orders_database.sqlite', 
                               table_name='forecast', 
                               if_exists="fail", 
                               **kwargs):
    
    # Prepared the data
    df = prep_forecast_data_for_update(data=data, 
                                       id_column=id_column, 
                                       date_column=date_column)
    
    # Check format for SQL Database
    
    # データ型の変換
    df['date'] = df['date'].dt.to_timestamp()
    
    df.info()
    
    # フィールド定義
    sql_dtype = {'id': String(), 
                 'date': String(), 
                 'value': Numeric(), 
                 'prediction': Numeric(), 
                 'ci_lo': Numeric(),
                 'ci_hi': Numeric()}
    
    # Connect to Database
    engine = sql.create_engine(conn_string)
    conn = engine.connect()
    
    # Make Table
    df.to_sql(con=conn, name=table_name, if_exists=if_exists, 
              dtype=sql_dtype, index=False
              #**kwargs
              )
    
    # Close Connection
    conn.close()
    
    pass


# 動作テスト
write_forecast_to_database(data=arima_forecast_df, 
                           id_column='category_2', 
                           date_column='order_date', 
                           if_exists='replace')


# 動作テスト
# --- テーブルを別名で作成
write_forecast_to_database(data=arima_forecast_df, 
                           id_column='category_2', 
                           date_column='order_date', 
                           table_name='forecast2', 
                           if_exists='replace')

# --- 不要テーブルの削除
conn_string='sqlite:///00_database/bike_orders_database.sqlite'
engine = sql.create_engine(conn_string)
conn = engine.connect()
sql.Table('forecast2', MetaData(conn)).drop()


# 3.0 READ FROM DATABASE --------------------------------------------

# 関数定義
def read_forecast_from_database(
    conn_string='sqlite:///00_database/bike_orders_database.sqlite', 
    table_name='forecast', 
    **kwargs
):
    
    # Connect to Database
    engine = sql.create_engine(conn_string)
    conn = engine.connect()
    
    # Read from table
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=conn, 
                     parse_dates=['date'])
    
    # Close Connection
    conn.close()
    
    return df


# 動作確認
# --- デフォルトでforecastテーブルを取得
# --- 別のテーブルも取得可能
read_forecast_from_database()
read_forecast_from_database(table_name='bikes')
read_forecast_from_database(table_name='forecast')
