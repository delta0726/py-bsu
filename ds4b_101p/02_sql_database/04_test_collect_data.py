# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : SQL DATABASES (Module 2): Working with SQLAlchemy
# File        : 04_test_collect_data.py
# Update Date : 2021/6/12
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - 自分で定義したパッケージを参照して関数を実行する


# ＜目次＞
# 1 動作確認


# 1 動作確認 ------------------------------------------------------------------------

# ライブラリ
from my_pandas_extensions.database import collect_data


# データ取得
# --- 引数省略も可能
collect_data(conn_string='sqlite:///00_database/bike_orders_database.sqlite')
collect_data()
