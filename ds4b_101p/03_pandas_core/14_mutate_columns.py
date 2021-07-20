# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : 集計列の追加
# File        : 15_mutate_columns.py
# Update Date : 2021/6/16
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 集計列の追加
# 2 列集計の事例(対数変換)
# 3 列集計の事例(フラグ追加)
# 4 列集計の事例(ビニング)
# 5 列集計の事例(範囲指定ビニング)
# 6 列集計の事例(データ型の変換)


# 0 準備 ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# データロード
df = pd.read_csv("00_data_raw/bike_agg_data.csv")
df


# 1 集計列の追加 ------------------------------------------------------------

# ＜ポイント＞
# - 集計列に対してPandas Seriesを追加する
# - 元データフレームのデータを使用するためラムダ式を使うことが多い

# データ確認
df.frame_material

# 列の変更
# --- 既存列の変更
# --- 小文字変換して列追加
df[['frame_material']].assign(frame_material=lambda x: x['frame_material'].str.lower())
df[['frame_material']].assign(frame_material_lower=lambda x: x['frame_material'].str.lower())


# 2 列集計の事例(対数変換) -----------------------------------------------------

# モデルごとの価格分布
# --- price
df[['model', 'price']]\
    .drop_duplicates()\
    .set_index('model')\
    .plot(kind='hist')

# プロット作成
plt.show()

# モデルごとの価格分布
# --- priceを対数変換
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price=lambda x: np.log(x['price']))\
    .set_index('model')\
    .plot(kind='hist')

# プロット作成
plt.show()


# 3 列集計の事例(フラグ追加) ----------------------------------------------------

# 文字列判定（find）
# --- 文字列が見つかると0を返す
# --- 文字列が見つからなければ-1を返す
'Supersix Evo Hi-Mod Team'.lower().find('supersix') >= 0
'Beast of the East 1'.lower().find('supersix') >= 0

# 文字列判定（contains）
# --- TRUE/FALSEを直接返す
df['model'].str.lower().str.contains('supersix')

# フラグの追加
# --- bool型で返される（指定した文字列ならTrue）
# --- 1/0にしたい場合は数値型に変換
df.assign(flag_supersix=lambda x: x['model'].str.lower().str.contains('supersix'))
df.assign(flag_supersix=lambda x: x['model'].str.lower().str.contains('supersix').astype(int))


# 4 列集計の事例(ビニング) ----------------------------------------------------

# Seriesでビニング適用
# --- CategoricalDtypeとして格納
# --- 必要に応じて文字列変換
bins = pd.cut(df.price, 3, labels=['low', 'median', 'high'])
bins.dtype
bins.astype('str')

# ビニング列の追加
df_bin = df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x: pd.cut(x.price, bins=3))

# 確認
# --- category型
df_bin
df_bin.info()

# 分位列の追加
# --- 文字列変換
df_tile = df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x: pd.cut(x.price, bins=3,
                                         labels=['low', 'median', 'high']).astype('str'))

# 確認
# --- category型
df_tile
df_tile.info()

# テーブル表示
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x: pd.cut(x.price, bins=3))\
    .pivot(index='model', columns='price_group', values='price')


# 5 列集計の事例(範囲指定ビニング) -------------------------------------------------

# 範囲指定でビニング
pd.qcut(df.price, q=[0, 0.33, 0.66, 1], labels=['low', 'medium', 'high'])

# テーブル表示
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x:pd.qcut(x.price, q=[0, 0.33, 0.66, 1]))\
    .pivot(index='model', columns='price_group', values='price')


# 6 列集計の事例(データ型の変換) -----------------------------------------------------

# 全列の変換
df.astype('str').info()

# 指定列の変換
# --- 複数列の同時指定が可能
df.astype({'price': str, 'total_price': str}).info()

# assginによる変換
# --- 1列のみ変更可能
df.assign(price=lambda x: (x.price.astype('str'))).info()
