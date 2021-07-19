# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 3 (Pandas Core): Data Structures
# File        : 01_data_structures_for_analysis.py
# Update Date : 2021/6/15
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - Pythonオブジェクトのデータ型の基礎概念を学ぶ


# ＜目次＞
# 0 準備
# 1 オブジェクトのクラス確認
# 2 DataFrameの構造
# 3 主なPythonオブジェクト
# 4 文字列の操作
# 5 低レベルオブジェクトから高レベルオブジェクトに変換
# 6 データ型の変換


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data


# データロード
df = collect_data()


# 1 オブジェクトのクラス確認 -------------------------------------------------

# データ型の確認
# --- pandas.core.frame.DataFrame
type(df)

# スーパークラスをすべて取得
type(df).mro()
type("string").mro()

# 属性データの確認
# --- 行列数
# --- 列名
df.shape
df.columns

# メソッドの実行
# --- オブジェクトはメソッドを持つ
# --- 例：query()
df.query("model == 'Jekyll Carbon 2'")


# 2 DataFrameの構造 -------------------------------------------------------

# ＜ポイント＞
# - データフレームは以下のような構造を取っている
#   --- PANDAS DATA FRAME
#   --- PANDAS SERIES
#   --- NUMPY

# PANDAS DATA FRAME
# --- pandas.core.series.Series,
# --- pandas.core.base.IndexOpsMixin,
# --- pandas.core.arraylike.OpsMixin,
# --- pandas.core.generic.NDFrame,
# --- pandas.core.base.PandasObject,
# --- pandas.core.accessor.DirNamesMixin,
# --- pandas.core.base.SelectionMixin,
# --- pandas.core.indexing.IndexingMixin
type(df)
type(df).mro()

# PANDAS SERIES
# --- pandas.core.series.Series,
# --- pandas.core.base.IndexOpsMixin,
# --- pandas.core.arraylike.OpsMixin,
# --- pandas.core.generic.NDFrame,
# --- pandas.core.base.PandasObject,
# --- pandas.core.accessor.DirNamesMixin,
# --- pandas.core.base.SelectionMixin,
# --- pandas.core.indexing.IndexingMixin,
type(df['order_date'])
type(df['order_date']).mro()


# NUMPY ARRAY
# --- [numpy.ndarray, object]
df['order_date'].values
type(df['order_date'].values)
type(df['order_date'].values).mro()


# 3 主なPythonオブジェクト ------------------------------------------------

# * 辞書 ---------------------------------------------

# ＜ポイント＞
# - df.remname()やdf.agg()で使うことが多い

# 定義
d = {'a': 1}

# 確認
type(d)
type(d).mro()

# 要素へのアクセス
d.keys()
d.values()

# データ取得
d['a']


# * リスト --------------------------------------------

# 定義
# --- 異なるデータ型を格納できる
lst = [1, "A", [2, "B"]]

# 確認
type(lst)
type(lst).mro()

# 要素へのアクセス
lst[0]
lst[1]
lst[2]


# * タプル ---------------------------------------------

# 定義
t = (10, 20)

# 確認
type(t)
type(t).mro()

# 確認
t[0]
t[1]


# * ベースデータ -------------------------------------------

# Float
type(1.5).mro()

# int
# --- numpyではint64を使用（スピードやメモリの改善のため）
type(1).mro()
df.total_price.values

# Character
# --- objectは文字列とカテゴリカルデータの総称
# --- Numpy配列は文字列ではない
# --- Numpy配列の各要素が文字列(str)となっている
df['model'].values
type(df['model'].values)
type(df['model'].values[0])


# 4 文字列の操作 ---------------------------------------------------------

# データ定義
model = "Jekyll Carbon 2"
price = 6070

# 文字列に変数を埋め込む
# --- f"{A}"で、文字列の中に変数を組み込むことができる
f"The first model is : {model}"
f"The price of first model : {price}"

# 文字列の結合
str(price) + " Some Text"

# 文字列の置換
"50%".replace("%", "")


# 5 低レベルオブジェクトから高レベルオブジェクトに変換 --------------------------

# ＜ポイント＞
# - リストから｢Numpy Array｣｢Pandas Series｣を作成する


# Rangeオブジェクト
# --- range()はジェネレータ
rng = range(1, 50)
type(rng).mro()

# リストに変換
r = list(range(1, 50))

# オブジェクト作成
# --- ndarray
# --- Pandas Series
np.array(r)
pd.Series(r)


# 6 データ型の変換 ----------------------------------------------------------

# ＜ポイント＞
# - データ型の確認はdtypesを用いる
# - データ型の変更はastype()を用いる

# データ型の確認
# --- Pandas DataFrame
# --- Pandas Series
df.dtypes
df['order_date'].dtypes

# 文字列に変換
df['order_date'].astype('str')
df['order_date'].astype('str').dtypes

# 文字列に変換
df['order_date'].astype('str').str.replace("-", "/")
