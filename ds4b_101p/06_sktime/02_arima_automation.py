# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 6 (Sktime): ARIMA Automation
# File        : 02_arima_automation.py
# Update Date : 2021/7/3
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************

# Imports

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time

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


# Sktime Imports

# FUNCTION DEVELOPMENT ----
# - arima_forecast(): Generates ARIMA forecasts for one or more time series.

def arima_forecast(data, h=12, sp=1, alpha=0.05,
                   suppress_warnings=True, *args, **kwargs):

    # データチェック

    # データ変換
    df = data

    # 結果格納用
    model_results_dict = {}

    # ループ処理
    for col in tqdm(df.columns, mininterval=0):

        # 系列の抽出
        y = df[col]

        # モデリング
        forecaster = AutoARIMA(sp=sp,
                               suppress_warnings=suppress_warnings,
                               # *args,
                               # **kwargs
                               )

        # 学習
        forecaster.fit(y)

        # 出力
        print(forecaster)

        # 予測
        # --- 予測値 + 信頼区間
        predictions, conf_int_df = \
            forecaster.predict(fh=np.arange(1, h+1),
                               return_pred_int=True,
                               alpha=alpha)

        # データフレームに格納
        ret = pd.concat([y, predictions, conf_int_df], axis=1)
        ret.columns = ["value", "prediction", "ci_lo", "ci_hi"]

        # 結果格納
        model_results_dict[col] = ret

    # データ結合
    model_results_df = pd.concat(model_results_dict, axis=0)
    
    # 列名追加
    nms = [*df.columns.names, *df.index.names]
    model_results_df.index.names = nms
    
    # インデックス削除
    ret = model_results_df.reset_index()

    # 不要行の削除
    cols_to_keep = ~ret.columns.str.startswith("level_")
    ret = ret.iloc[:, cols_to_keep]

    return ret


# データ格納
data = bike_sales_cat2_m_df

# 関数実行
fcast = arima_forecast(data, h=12, sp=1)

# 動作テスト
# --- 単一系列
# --- 複数系列
arima_forecast(bike_sales_m_df, h=12, sp=1)
arima_forecast(bike_sales_cat2_m_df, h=12, sp=1)
