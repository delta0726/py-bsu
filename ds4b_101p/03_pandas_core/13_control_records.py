# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : 行操作の方法を学ぶ
# File        : 14_control_records.py
# Update Date : 2021/6/16
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 列要素のユニーク化
# 2 並び替え
# 3 フィルタリング(ブラケット)
# 4 フィルタリング(loc)
# 5 フィルタリング(query)
# 6 フィルタリング(isin)
# 7 スライシング
# 8 特定列のトップボトムの抽出
# 9 レコードのユニーク化
# 10 サンプリング


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd


# データロード
df = pd.read_csv("00_data_raw/bike_agg_data.csv")
df


# 1 列要素のユニーク化 ---------------------------------------------

# ＜ポイント＞
# - レコード操作の前に系列レベルでデータを確認することが多い


# シリーズのユニーク化
df['category_2'].unique()

# カテゴリの要素数
df['category_2'].value_counts()


# 2 並び替え ----------------------------------------------------

# データフレームのソート
# --- デフォルトはascending=True
df[['total_price']].sort_values('total_price')
df[['total_price']].sort_values('total_price', ascending=False)

# シリーズのソート
df['price'].sort_values()
df['price'].sort_values(ascending=False)

# インデックスのソート
df.set_index('city').sort_index(axis=0)
df.set_index('city').sort_index(axis=0, ascending=False)


# 3 フィルタリング(ブラケット)  ----------------------------------------

# Seriesレベルの動作
# --- 演算子を適用すると大小関係でTrue/Falseで表現する
df.order_date >= pd.to_datetime("2015-01-01")

# フィルタリング
df[df.order_date >= pd.to_datetime("2015-01-01")]

# 文字列判定フィルタリング
# --- 完全一致フィルタ
# --- 先頭一致フィルタ
# --- containフィルタ
# --- 正規表現フィルタ
df[df.model == 'Trigger Carbon 1']
df[df.model.str.startswith('Trigger')]
df[df.model.str.contains('Carbon')]
df[df.model.str.match('^C')]

# NOTフィルタ
# --- ~でTRUE/FALSEを反転させる
df[~df.model.str.contains('Carbon')]

# 複数条件フィルタ
# --- AND条件
# --- OR条件
df[(df.order_date >= pd.to_datetime("2015-01-01")) & (df.model.str.contains('Carbon'))]
df[(df.order_date >= pd.to_datetime("2015-01-01")) | (df.model.str.contains('Carbon'))]


# フィルタリング(loc)  ----------------------------------------------

# フィルタリング
df.loc[lambda x: x.order_date >= pd.to_datetime("2015-01-01")]

# 文字列判定フィルタリング
# --- 完全一致フィルタ
# --- 先頭一致フィルタ
# --- containフィルタ
# --- 正規表現フィルタ
df.loc[lambda x: x.model == 'Trigger Carbon 1']
df.loc[lambda x: x.model.str.startswith('Trigger')]
df.loc[lambda x: x.model.str.contains('Carbon')]
df.loc[lambda x: x.model.str.match('^C')]

# NOTフィルタ
# --- ~でTRUE/FALSEを反転させる
df.loc[lambda x: ~x.model.str.contains('Carbon')]

# 複数条件フィルタ
# --- AND条件
# --- OR条件
df.loc[lambda x: (x.order_date >= pd.to_datetime("2015-01-01")) &
                 (x.model.str.contains('Carbon'))]

df.loc[lambda x: (x.order_date >= pd.to_datetime("2015-01-01")) |
                 (x.model.str.contains('Carbon'))]


# 5 フィルタリング(query) ------------------------------------------

# フィルタ条件を文字列で指定
# --- @を付けることで、文字列の中に変数を呼び込むことができる
price_threshold_1 = 5000
price_threshold_2 = 1000
df.query('price >= @price_threshold_1')
df.query('(price >= @price_threshold_1) | (price <= @price_threshold_2)')

# フィルタ条件をfstringで指定
df.query(f"price >= {price_threshold_1}")


# 6 フィルタリング(isin) -------------------------------------------

# ＜ポイント＞
# - isin()は、指定した値の有無でTRUE/FALSEのbool型を出力する
# - 複数条件のOR指定で出力する場合にスッキリ記述することができる

# Seriesレベルの動作
# --- isinメソッドで指定した文字列と一致するかでTrue/False
# --- ~でTrue/Falseを反転させることも可能
df['category_2'].isin(['Triathalon', 'Over Mountain'])
~df['category_2'].isin(['Triathalon', 'Over Mountain'])

# フィルタ抽出
df[df['category_2'].isin(['Triathalon', 'Over Mountain'])]

# locを用いた抽出
df.loc[lambda x: x.category_2.isin(['Triathalon', 'Over Mountain'])]


# 7 スライシング -------------------------------------------------------

# ＜ポイント＞
# - スライシングはインデックスに基づいて行うフィルタリング


# ブラケットで抽出
df[:5]

# 先頭/末尾の抽出
# --- この操作もインデックスを用いて行われている
df.head(5)
df.tail(5)

# 行のスライシング
# --- ilocはインデックス抽出用
# --- ｢:｣で全体を取得
df.iloc[0:5, [1, 3, 5]]
df.iloc[0:5, :]
df.iloc[:, [1, 3, 5]]

# 複数行の連続スライシング
# --- 範囲が非連続の場合はnumpyの｢r_｣を使うことで実現することができる
df.iloc[np.r_[0:10, 20:30]]
df.iloc[np.r_[0:10, 20:30], :]


# 8 特定列のトップボトムの抽出 ---------------------------------------------

# ＜ポイント＞
# - 特定列の水準の上位又は下位を抽出する
#   --- head()/tail()はインデックスを用いて抽出している


# Seriesレベルの動作
df['total_price'].nlargest(n=20)
df['total_price'].nsmallest(n=20)

# データフレームによる抽出
# --- 指定列の上位/下位による抽出
df.nlargest(n=20, columns='total_price')
df.nsmallest(n=20, columns='total_price')


# 9 レコードのユニーク化 -------------------------------------------------

# 単一列のユニーク化
# --- Pandas Dataframeに適用
df[['model']].drop_duplicates()

# 複数列のユニーク化
df[['model', 'category_1', 'category_2', 'frame_material']].drop_duplicates()


# 10 サンプリング -------------------------------------------------------

# レコードのサンプリング
# --- 行数指定
# --- パーセント指定
df.sample(n=10, random_state=123)
df.sample(frac=0.1, random_state=123)
