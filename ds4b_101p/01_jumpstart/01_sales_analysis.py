# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : (Module 1): First Sales Analysis with Python
# File        : 01_sales_analysis.py
# Update Date : 2021/6/10
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - プロジェクトのフローを簡単に確認する


# ＜備考＞
# - PlotineはPycharmでは正常に動作しないのでVScodeのJupyterを使う
#   --- ｢重ね書き｣が個別プロットとして認識されるようだ


# ＜目次＞
# 1 ライブラリのロード
# 2 データのインポート
# 3 データを調べる
# 4 初めてのメソッドチェーン
# 5 データ結合
# 6 コピーと参照
# 7 データ操作
# 8 データ可視化


# 1 ライブラリのロード ----------------------------------------------------

# Core Python Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plotting
from plotnine import (
    ggplot, aes, 
    geom_line, geom_smooth,
    facet_wrap, 
    scale_y_continuous,
    scale_x_datetime, 
    labs, 
    theme, theme_minimal,
    expand_limits, element_text
)

from mizani.breaks import date_breaks
from mizani.formatters import date_format, currency_format

from rich import pretty
pretty.install()


# 2 データのインポート ---------------------------------------------------

# ＜パスの取得方法＞
# - 1. "/" を入力するとオートコンプリートが発動する
# - 2. パスを全て入力したら、先頭の/を消す


# データ1
bikes_df = pd.read_excel("00_data_raw/bikes.xlsx")
bikes_df

# データ2
bikeshops_df = pd.read_excel("00_data_raw/bikeshops.xlsx")
bikeshops_df

# データ3
# --- そのままインポート
orderline_df = pd.read_excel("00_data_raw/orderlines.xlsx")

# データ3
# --- 日付を文字列としてインポート
orderline_df = pd.read_excel(io="00_data_raw/orderlines.xlsx",
                             converters={'order.date': str})
orderline_df


# 3 データを調べる -------------------------------------------------------

# データ概要
bikes_df.info()
bikeshops_df.info()
orderline_df.info()

# 先頭データ確認
# --- データフレームの先頭データ確認はheadメソッドを使う
bikes_df.head(10)
orderline_df.head(10)
bikeshops_df.head(10)

# 列の先頭データの確認
# --- ブラケットを用いて抽出
bikes_df.price[0:5]


# 4 初めてのメソッドチェーン ----------------------------------------------

# ＜ポイント＞
# - Pandasはメソッドチェーンによる記述で、コードの見通しがよくなる


# 逐次的に記述
# --- 列の取得
# --- アイテムごとの頻度を計算
# --- 上位5つの値を取得
s = bikes_df['description']
freq_count_series = s.value_counts()
freq_count_series.nlargest(5)

# メソッドチェーンで記述
# --- ブラケットでデータを取得
top5_bikes_series = bikes_df['description']\
    .value_counts()\
    .nlargest(5)

# プロット作成
# --- Pandas Seriesのメソッドを使用
fig = top5_bikes_series.plot(kind="barh")
fig.invert_yaxis
plt.show()


# 5 データ結合 --------------------------------------------------------

# データ確認
orderline_df.head()
bikes_df.head()

# テーブル結合
# --- 列の削除（Unnamed: 0）
# --- Left Joinで結合
bike_orderlines_joined_df = orderline_df\
    .drop(columns='Unnamed: 0', axis=1)\
    .merge(right=bikes_df, how='left', left_on='product.id', right_on='bike.id')\
    .merge(right=bikeshops_df, how='left', left_on='customer.id', right_on='bikeshop.id')

# データ確認
bike_orderlines_joined_df

# 列の確認
orderline_df.drop(columns='Unnamed: 0', axis=1).info()
bikes_df.info()
bikeshops_df.info()
bike_orderlines_joined_df.info()


# 6 コピーと参照 ------------------------------------------------------

# ＜ポイント＞
# - コピーと参照(代入)は挙動が異なるので要注意
#   --- 元データを加工する(元データを保存しない)のが大半なので｢参照｣を用いることが多い


# * 準備：2つのオブジェクトを作成 ----------------------

# 参照
# --- 変更は元のオブジェクトに反映
df = bike_orderlines_joined_df

# コピー
# --- メモリに新しいオブジェクトを作成（変更は反映しない）
df2 = bike_orderlines_joined_df.copy()


# * 実験：データが型の変更の反映確認 ----------------------

# データ確認
# --- 'order.date'は文字列(object)
df.info()
df2.info()

# 日付型に変換
# --- 元データは文字列
df['order.date'] = pd.to_datetime(df['order.date'])

# 結果
# --- 参照  ：データ変更が反映されている
# --- コピー：データ変更が反映されていない（元のデータが保存されている）
df.info()
df2.info()


# 7 データ加工 -----------------------------------------------------------

# ＜ポイント＞
# - 以降で使用するプロット用のデータ加工を行っている
#   --- データ加工の説明は本編で行われる
#   --- メソッドチェーンの活用も重視してない


# * Seriesの取得 -------------------------------------

# 列参照
# --- .列名で指定（インテリセンスが発動する）
df.description
df.location

# 転置
df.T


# * descriptionの分割 --------------------------------

# 文字列の分割
# --- 分割パターンがある文字列
# --- 文字列を分割してリスト表示
"Mountain - Over Mountain - Carbon"
"Mountain - Over Mountain - Carbon".split(" - ")

# 列の分割
# --- expnad引数：True(列に分割してDataFrameで出力) False(リストのままSeries出力)
# --- 列名はついていないので、(0 ,1, 2)で表示
temp_df = df['description'].str.split(pat=' - ', expand=True)
temp_df

# 元のデータフレームに追加
df['category.1'] = temp_df[0]
df['category.2'] = temp_df[1]
df['frame.material'] = temp_df[2]

# 確認
df.info()


# * locationの分割 ----------------------------------

# 列の分割
temp_df = df['location'].str.split(', ', n=-1, expand=True)
df['city'] = temp_df[0]
df['state'] = temp_df[1]

# 確認
df.info()


# * total.priceの作成 ------------------------------

# 列を計算して追加
df['total.price'] = df['quantity'] * df['price']

# 確認
df.info()
df.sort_values('total.price', ascending=True)
df.sort_values('total.price', ascending=False)


# * 列を取得する -----------------------------------

# 列の確認
# --- 出力結果がリストになっているので、列名を一括コピーするのに便利
df.columns

# 列の抽出
# --- df.columnsの出力結果からリストを取得
# --- quantityはpriceの後ろに移動
cols_to_keep_list = [
    'order.id', 
    'order.line', 
    'order.date', 
    # 'customer.id', 
    # 'product.id',
    'bike.id', 
    'model', 
    # 'description', 
    'price', 
    'quantity', 
    # 'bikeshop.id',
    'bikeshop.name', 
    'location', 
    'category.1', 
    'category.2',
    'frame.material', 
    'city', 
    'state', 
    'total.price'
    ]

# データ出力
# --- 変数がリストなので[]で選択
df = df[cols_to_keep_list]


# * 列名の変更 ----------------------------------------------

# 文字列の置換
# --- 文字列はstrクラスのオブジェクト、replaceはメソッド
# --- 仮にヘルプ参照したい場合は次のように指定（?str.replace）
'order.date'.replace(".", "_")

# 列名を抽出して置換
# --- 複数列を同時置換
df.columns = df.columns.str.replace(".", "_")

# 確認
df.columns

# データ格納
bike_orderlines_wrangle_df = df
bike_orderlines_wrangle_df

# 元データと比較
bike_orderlines_joined_df

# データ保存
# mkdir("00_data_wrangled")
# bike_orderlines_wrangle_df.to_pickle("00_data_wrangled/bike_orderlines_wrangled_df.pkl")
# df = pd.read_pickle("00_data_wrangled/bike_orderlines_wrangled_df.pkl")


# 8 データ可視化 -------------------------------------------------------------------

# 8.1 月ごとの売上高の集計（グループ集計なし）----------------

# * データ準備 ------------------------------

# 日付データの変換
order_date_series = df['order_date']
order_date_series.dt.year

# データ集計
# --- 日付集計するため、一旦日付をインデックスとして設定(集計後に解除)
# --- リサンプルとはデータ頻度を変える操作（rule: MS(Month Start)）
# --- sum() ≒ aggregate(np.sum)
sales_by_month_df = df[['order_date', 'total_price']]\
    .set_index('order_date')\
    .resample(rule='MS')\
    .aggregate(np.sum)\
    .reset_index()

# データ確認
sales_by_month_df


# * Quick Plot ----------------------------

# プロット作成
# --- {matplotlib}の簡易的なプロット
sales_by_month_df.plot(x='order_date', y='total_price')
plt.show()


# Report Plot ----------------------------

# 通貨フォーマット
# --- {mizani}
usd = currency_format(prefix='$', digits=0, big_mark=',')
usd([1000])

# プロット作成
# --- {plotnine}でggplot2ベースのプロット
ggplot(aes(x='order_date', y='total_price'), data=sales_by_month_df) + \
    geom_line() + \
    geom_smooth(method='loess', se=False, color='blue', span=0.3) + \
    scale_y_continuous(labels=usd) + \
    labs(title='Revenue by Month', x='', y='Revenue') + \
    theme_minimal() + \
    expand_limits(y=0)


# 8.2 月ごとの売上高の集計（グループ集計あり）----------------

# ** Step 1 - Manipulate ------------

# データ加工
# --- グループごとに合計
sales_by_month_cat_2 = df[['category_2', 'order_date', 'total_price']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('W')\
    .agg(func={'total_price': np.sum})\
    .reset_index()

# データ確認
sales_by_month_cat_2


# Step 2 - Visualize ----

# Simple Plot

# プロット作成
# --- {matplotlib}
sales_by_month_cat_2\
    .pivot(index='order_date', columns='category_2', values='total_price')\
    .fillna(0)\
    .plot(kind='line', subplots=True, layout=(3, 3))

# プロット表示
plt.show()


# Reporting Plot

# 通貨フォーマット
# --- {mizani}
usd = currency_format(prefix='$', digits=0, big_mark=',')

# プロット作成
# --- {plotnine}
ggplot(aes(x='order_date', y='total_price'), data=sales_by_month_cat_2) + \
    geom_line(color='#2c3e50') + \
    geom_smooth(method='lm', se=False, color='blue') + \
    facet_wrap(facets='category_2', ncol=3, scales='free_y') + \
    scale_y_continuous(labels=usd, size=2) + \
    scale_x_datetime(
        breaks=date_breaks('2 years'), 
        labels=date_format(fmt='%y-%m')) + \
    labs(title='Revenue by week', x='', y='Revenue') + \
    theme_minimal() + \
    theme(
        subplots_adjust={'wspace': 0.35},
        axis_text_y=element_text(size=6), 
        axis_text_x=element_text(size=6)
        )
