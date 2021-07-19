# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 2 (Pandas Import): Importing Files
# File        : 01_import_files.py
# Update Date : 2021/6/12
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - データロードの方法を学ぶ
#   --- Pandasは主なデータをロードする関数を備えている


# ＜目次＞
# 0 準備
# 1 パスの取得
# 2 Pickleファイル
# 3 CSVファイル
# 4 Excelファイル


# 0 準備 ------------------------------------------------------------------

import os
import pandas as pd


# カレントパスの確認
os.getcwd()


# 1 パスの取得 ---------------------------------------------------------------

# ＜VScode＞
# - 1. "/" を入力するとオートコンプリートが発動する
# - 2. パスを全て入力したら、先頭の/を消す

# ＜Pycharm＞
# - 1. pd.read_csv("")で、""を入力した時点でオートコンプリートが発動する（作業ファイルのパスを基準）
# - 2. "../"などと入力することで、上のディレクトリに移動
#      例： pd.read_csv("../00_data_wrangled/bike_orderlines_wrangled_df.csv")
# - 3. このままだとエラーになるので、"../"を消しておく
#      例： pd.read_csv("00_data_wrangled/bike_orderlines_wrangled_df.csv")


# 2 Pickleファイル ---------------------------------------------------------

# ＜ポイント＞
# - Pickleファイルとは、Pythonオブジェクトをファイル保存する際のファイル形式

# データロード
pickle_df = pd.read_pickle("00_data_wrangled/bike_orderlines_wrangled_df.pkl")

# データ確認
pickle_df.info()


# 3 CSVファイル ---------------------------------------------------------------

# インポート
# --- 数値か文字列としてデータを格納
# --- 日付データとしてインポートするには引数で指定が必要
csv_df = pd.read_csv("00_data_wrangled/bike_orderlines_wrangled_df.csv")
csv_df.info()

# インポート
# --- 数値か文字列としてデータを格納
# --- 日付データとしてインポートするには引数で指定が必要
csv_df = pd.read_csv("00_data_wrangled/bike_orderlines_wrangled_df.csv", parse_dates=['order_date'])
csv_df.info()


# 4 Excelファイル ------------------------------------------------------------

# インポート
excel_df = pd.read_excel("00_data_wrangled/bike_orderlines_wrangled_df.xlsx")

# データ確認
excel_df.info()
