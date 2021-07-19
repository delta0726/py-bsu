# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Pandasデータフレームの基本操作
# File        : 11_pandas_basic.py
# Update Date : 2021/6/16
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 サイズ取得
# 2 行と列の情報
# 3 データ確認
# 4 系列のユニーク化


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import pandas as pd

from my_pandas_extensions.database import collect_data


# データロード
df = collect_data()
df


# 1 サイズ取得 ------------------------------------------------------------

# 行列数
df.shape

# 行数
df.shape[0]
len(df)

# 列数
df.shape[1]
len(df.columns)

# 全要素数
df.size


# 2 行と列の情報 ----------------------------------------------------------

# 列の取得
# --- 列名をリストで取得することができる（利用頻度が多い）
df.columns

# インデックスの取得
df.index


# 3 データ確認 -------------------------------------------------------------

# 先頭/末尾の抽出
df.head()
df.tail()

# サマリー
df.info()

# 列ごとのデータ型
df.dtypes

# 基本統計量
df.describe()

# 相関係数行列
df.select_dtypes(include='int64').corr()


# 4 系列のユニーク化----------------------------------------------------------

# 系列のユニーク化
df['category_2'].unique()

# ユニーク要素のカウント
df['category_2'].value_counts()
