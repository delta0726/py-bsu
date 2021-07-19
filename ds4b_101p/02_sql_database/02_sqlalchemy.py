# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : SQL DATABASES (Module 2): Working with SQLAlchemy
# File        : 02_sqlalchemy.py
# Update Date : 2021/6/12
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - {sqlalchemy}を使ってSQliteのデータベースを操作する
# - SQLによるデータ取得方法を学ぶ


# ＜目次＞
# 0 準備
# 1 データロード
# 2 DB作成
# 3 DBの基本操作


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import pandas as pd
import sqlalchemy as sql


# 1 データロード -----------------------------------------------------------

# データロード
bikes_df = pd.read_excel("00_data_raw/bikes.xlsx")
bikeshops_df = pd.read_excel("00_data_raw/bikeshops.xlsx")
orderlines_df = pd.read_excel("00_data_raw/orderlines.xlsx")


# 2 DB作成 ----------------------------------------------------------------

# DB接続
# --- DB定義
# --- 接続の確立(DBがない場合はファイル作成)
engine = sql.create_engine("sqlite:///00_database/bike_orders_database.sqlite")
conn = engine.connect()

# テーブル作成
# --- エクセルデータをそのまま登録
bikes_df.to_sql("bikes", con=conn, if_exists='replace')
bikeshops_df.to_sql("bikeshops", con=conn, if_exists='replace')
orderlines_df.iloc[:, 1:].to_sql("orderlines", con=conn, if_exists='replace')

# データ取得
pd.read_sql("SELECT * FROM bikes", con=conn)
pd.read_sql("SELECT * FROM bikeshops", con=conn)
pd.read_sql("SELECT * FROM orderlines", con=conn)

# DB接続の解放
conn.close()


# 3 DBの基本操作 -----------------------------------------------------------

# DB接続
engine = sql.create_engine("sqlite:///00_database/bike_orders_database.sqlite")
conn = engine.connect()

# テーブル名の取得
engine.table_names()

# スキーマの取得
inspector = sql.inspect(conn)
inspector.get_schema_names()
inspector.get_table_names('main')
inspector.get_table_names()

# データ取得
# --- テーブル名を全て取得
# --- SQLでデータ取得（テーブル名を変数化）
table = inspector.get_table_names()
pd.read_sql(f"SELECT * FROM {table[0]}", con=conn)

# DB解放
conn.close()
