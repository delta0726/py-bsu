# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 6 (Sktime): Introduction to Forecasting
# File        : 01_sktime_forecast.py
# Update Date : 2021/7/2
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - {sktime}のArimaモデルを確認する
# - ループ処理による複数モデルの実行を行う


# ＜目次＞
# 0 準備
# 1 データ加工
# 2 単一系列の時系列予測
# 3 複数系列の時系列予測


# 0 準備 ------------------------------------------------------------------------

# Imports

# * Core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# * Original
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time

# * sktime
from sktime.forecasting.arima import AutoARIMA
from sktime.utils.plotting import plot_series

# * Progress Bar
from tqdm import tqdm


# データロード
df = collect_data()

# ヘルプ確認
# ?AutoARIMA
# ?summarize_by_time


# 1 データ加工 ------------------------------------------------------------------

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


# 2 単一系列の時系列予測 ---------------------------------------------------------

# プロット作成
# --- 単一系列
bike_sales_m_df.plot()
plt.show()

# データ格納
# --- Pandas Series
y = bike_sales_m_df['total_price']

# インスタンス生成
# --- sp: Seasonal Period
forecaster = AutoARIMA(sp=12)

# 学習
forecaster.fit(y)

# 予測
# --- 予測のみ
h = 24
forecaster.predict(fh=np.arange(1, h+1))

# 予測+信頼区間
# --- タプルでそれぞれ格納
prediction_ci_tuple = forecaster.predict(fh=np.arange(1, h+1),
                                         return_pred_int=True,
                                         alpha=0.95)

# クラス確認
type(prediction_ci_tuple)

# データ確認
prediction_ci_tuple[0]
prediction_ci_tuple[1]

# 予測+信頼区間
# --- タプルの中身を別けて変数に格納
predictions_series, conf_inf_df = \
    forecaster.predict(fh=np.arange(1, h+1),
                       return_pred_int=True,
                       alpha=0.95)

# データ確認
predictions_series
conf_inf_df

# プロット作成
plot_series(y,
            predictions_series,
            conf_inf_df['lower'],
            conf_inf_df['upper'],
            labels=['actual', 'prediction', 'upper_ci', 'lower_ci'])
plt.show()


# 3 複数系列の時系列予測 ---------------------------------------------------------

# 準備 ---------------------------------------

# データ確認
bike_sales_cat2_m_df.head()
bike_sales_cat2_m_df.info()

# データ格納
df = bike_sales_cat2_m_df

# データ抽出
df[df.columns[1]]


# ループ処理の確認 ------------------------------

# ループ処理1
# --- 列名の抽出
for col in df.columns:
    print(col)

# ループ処理2
# --- データの抽出
for col in df.columns:
    y = df[col]
    print(y[0:5])


# ARIMAモデルのループ処理 ------------------------

# 辞書作成
# --- 結果格納用
model_results_dict = {}

# ループ処理
for col in tqdm(df.columns[0:3]):
    # 系列の抽出
    y = df[col]

    # インスタンス生成
    # --- sp=12
    # --- spを小さくすると計算が速くなる
    forecaster = AutoARIMA(sp=12, suppress_warnings=True)

    # 学習
    forecaster.fit(y)

    # 予測
    h = 12
    predictions, conf_int_df = \
        forecaster.predict(fh=np.arange(1, h+1),
                           return_pred_int=True,
                           alpha=0.05)

    # データ結合
    ret = pd.concat([y, predictions, conf_int_df], axis=1)
    ret.columns = ["value", "prediction", "ci_low", "ci_hi"]

    # 結果格納
    model_results_dict[col] = ret

    # 結果出力
    # print(ret)


# 結果確認
model_results_dict.keys()
model_results_dict[('total_price', 'Cross Country Race')]

# データ結合
model_results_df = pd.concat(model_results_dict, axis=0)

# プロット作成
model_results_dict[list(model_results_dict.keys())[1]].plot()
plt.show()
