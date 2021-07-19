# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Data Wrangling
# File        : 02_data_wrangling.py
# Update Date : 2021/6/16
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - Pandasデータフレームの操作方法を学ぶ


# ＜目次＞
# 0 準備
# 1 基本属性の取得
# 2 列の選択/削除
# 3 列の並び替え


# 0 準備 ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_pandas_extensions.database import collect_data


# データロード
df = collect_data()
df


# 1 基本属性の取得 -----------------------------------------------------------

# ＜トピック＞
# 1-1 データフレームのサイズ
# 1-2 行と列の情報
# 1-3 系列情報


# 1-1 データフレームのサイズ ------------------------------------

# サイズ
# --- 全要素数
# --- 行列数
df.size
df.shape

# 行数
df.shape[0]
len(df)

# 列数
df.shape[1]
len(df.columns)


# 1-2 行と列の情報 --------------------------------------------

# 列の取得
df.columns

# インデックスの取得
df.index


# 1-3 系列情報 -----------------------------------------------

# サマリー
df.info()

# 列ごとのデータ型
df.dtypes

# 基本統計量
df.describe()


# 2 列の選択/削除 ---------------------------------------------------------------------

# ＜トピック＞
# 2-1 名前で列抽出
# 2-2 Seriesの抽出
# 2-3 列番号で列抽出
# 2-4 正規表現で列抽出
# 2-5 列の削除

# 2-1 名前で列抽出 ------------------------------------

# 複数列の抽出
# --- ブラケット[]で直接取得する
# --- []の中にリストを与える
df[['order_date', 'order_id', 'order_line']]

# 複数列の抽出
# - locプロパティを使用する（行/列を両方指定）
# - データフレーム全体を抽出
df.loc[:, ['order_date', 'order_id', 'order_line']]
df.loc[:, :]


# 2-2 Seriesの抽出 --------------------------------------

# Seriesとして抽出
# --- Pandas Seriesとして抽出 (psr1 / psr2)
# --- Pandas DataFrameとして抽出 (pdf)
psr1 = df['order_date']
psr2 = df.order_date
type(psr1)
type(psr2)

# データフレームとして抽出
pdf = df[['order_date']]
type(pdf)


# 2-3 列番号で列抽出 --------------------------------------

# 複数列の抽出
# --- 列番号を指定して抽出
# --- データフレーム全体を抽出
# --- 最後の3列を抽出
df.iloc[:, 0:3]
df.iloc[:, :]
df.iloc[:, -3:]


# 2-4 正規表現で列抽出 -----------------------------------

# 正規表現で列抽出
# --- filter()は主に正規表現での抽出に使用する
# --- ^model: starts_with("model")
# --- price$: ends_with("price")
df.filter(regex="(^model)|(^cat)", axis=1)
df.filter(regex="(price$)|(date$)", axis=1)

# 文字列を選択して列抽出
# --- 列名を抽出して文字列を判定
df.loc[:, lambda x: df.columns.str.startswith(("model", "cat"))]
df.loc[:, lambda x: df.columns.str.endswith(("price", "date"))]


# 2-5 列の削除 -----------------------------------------

# 列の削除
df.drop(['model', 'category_1', 'category_2'], axis=1)


# 3 列の並び替え ---------------------------------------------------------------------

# ＜トピック＞
# 3-1 特定列を先頭にする（単一列）
# 3-2 複数の特定列を先頭する（単純な方法）
# 3-3 複数の特定列を先頭する（内包表記）
# 3-4 複数の特定列を先頭する（concatを活用）
# 3-5 データ型で並び替え


# 3-1 特定列を先頭にする（単一列）--------------------------

# 単一列のソート
# --- tolist()でインデックスをリストに変換
# --- *でリストをアンパック
col_list = df.columns.tolist()
col_list.remove('model')
df[['model', *col_list]]

# 参考：アンパックの動作
# --- アンパックなし（lのリストが維持される）
# --- アンパックあり（lのリストが解除される）
['model', col_list]
['model', *col_list]


# 3-2 複数の特定列を先頭する（単純な方法）-----------------------

# 複数列のソート
col_lst = df.columns.tolist()
col_lst.remove('model')
col_lst.remove('category_1')
col_lst.remove('category_2')
df[['model', 'category_1', 'category_2', *col_lst]]


# 3-3 複数の特定列を先頭する（内包表記）-----------------------

# 複数列のソート
col_lst = df.columns.tolist()
cols_to_front = ['model', 'category_1', 'category_2']
l2 = [col for col in col_lst if col not in cols_to_front]
df[[*cols_to_front, *l2]]


# 3-4 複数の特定列を先頭する（concatを活用）--------------------

# 複数列のソート
df1 = df[['model', 'category_1', 'category_2']]
df2 = df.drop(['model', 'category_1', 'category_2'], axis=1)
pd.concat([df1, df2], axis=1)
df.info()


# 3-5 複数の特定列を先頭する（データ型に着目）--------------------

# 複数列のソート
# --- 先頭行にするのが文字列(object)であることに着目
df1 = df.select_dtypes(include='object')
df2 = df.select_dtypes(exclude='object')
pd.concat([df1, df2], axis=1)
df.info()


# 4 行の操作 ---------------------------------------------------------------------

# ＜トピック＞
# 4-1 並び替え
# 4.2 フィルタリング


# 4-1 並び替え -----------------------------------------------------

# データの並び替え
# --- データフレーム
# --- デフォルトはascending=True
df[['total_price']].sort_values('total_price')
df[['total_price']].sort_values('total_price', ascending=False)

# 日付の並び替え
# --- データフレーム
df[['order_date']].sort_values('order_date')
df[['order_date']].sort_values('order_date', ascending=False)

# データの並び替え
# --- Pandas Series
df.price.sort_values()
df['price'].sort_values()
df['price'].sort_values(ascending=False)


# 4.2 フィルタリング  --------------------------------------------------

# 不等号フィルタ
# --- Seriesレベルのフィルタ
# --- 大小関係でTrue/Falseに変換される
df.order_date >= pd.to_datetime("2015-01-01")
df[df.order_date >= pd.to_datetime("2015-01-01")]

# 完全一致フィルタ
df.model == 'Trigger Carbon 1'
df[df.model == 'Trigger Carbon 1']

# 先頭一致フィルタ
df.model.str.startswith('Trigger')
df[df.model.str.startswith('Trigger')]

# containフィルタ
df.model.str.contains('Carbon')
df[df.model.str.contains('Carbon')]


# クエリ

# フィルタ条件を文字列で指定
# --- @を付けることで、文字列の中に変数を呼び込むことができる
# --- fstringを用いて指定することも可能（推奨）
price_threshold_1 = 5000
price_threshold_2 = 1000
df.query('price >= @price_threshold_1')
df.query('(price >= @price_threshold_1) | (price <= @price_threshold_2)')
df.query(f"price >= {price_threshold_1}")


# Filtering Items in a List

# カテゴリ確認
# --- カテゴリ列挙
# --- カテゴリカウント
df['category_2'].unique()
df['category_2'].value_counts()

# Seriesアイテムを指定して抽出
# --- isinメソッドで指定した文字列と一致するかでTrue/False
# --- ~でTrue/Falseを反転させることも可能
df['category_2'].isin(['Triathalon', 'Over Mountain'])
~df['category_2'].isin(['Triathalon', 'Over Mountain'])
df[df['category_2'].isin(['Triathalon', 'Over Mountain'])]


# Slicing

df[:5]
df.head(5)
df.tail(5)


# インデックス・スライシング
# --- ilocはインデックス抽出用
# --- ｢:｣で全体を取得
df.iloc[0:5, [1, 3, 5]]
df.iloc[0:5, :]
df.iloc[:, [1, 3, 5]]


# 複数列のユニーク化
# --- Pandas Dataframeに適用
df[['model', 'category_1', 'category_2', 'frame_material']]\
    .drop_duplicates()


# 単一列のユニーク化
# --- Pandas Seriesに適用
df['model'].unique()
df.model.unique()


# 特定列の水準による抽出
# --- Pandas Dataframeに適用
df.nlargest(n=20, columns='total_price')
df.nsmallest(n=20, columns='total_price')


# 特定列の水準による抽出
# --- Pandas Seriesに適用
df['total_price'].nlargest(n=20)
df['total_price'].nsmallest(n=20)
df.total_price.nlargest(n=20)
df.total_price.nsmallest(n=20)


# レコードのサンプリング
# --- 行数指定
# --- パーセント指定
df.sample(n=10, random_state=123)
df.sample(frac=0.1, random_state=123)


# 4.0 ADDING CALCULATED COLUMNS (MUTATING) ----------------------------

# Method 1 - Series Notations --------------------------------

df2 = df.copy()

# 列の追加
df2['new_col'] = df2['price'] * df2['quantity']
df2['new_col_2'] = df['model'].str.lower()


# Method 2 - assign ------------------------------------------

# データ確認
df.frame_material

# 列の変更
# --- 小文字変換
# --- 小文字変換して列追加
df.assign(frame_material = lambda x: x['frame_material'].str.lower())
df.assign(frame_material_lower = lambda x: x['frame_material'].str.lower())


# 対数変換 ----------------------------------------------------

# データ分布の確認
# --- price
df[['model', 'price']]\
    .drop_duplicates()\
    .set_index('model')\
    .plot(kind='hist')

# データ分布の確認
# --- priceを対数変換
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price=lambda x: np.log(x['price']))\
    .set_index('model')\
    .plot(kind='hist')


# 文字列判定
'Supersix Evo Hi-Mod Team'.lower().find('supersix') >= 0
'Beast of the East 1'.lower().find('supersix') >= 0
df['model'].str.lower().str.contains('supersix')

# Adding Flags (True/False)
df\
    .assign(flag_supersix=lambda x: x['model']
            .str.lower().str.contains('supersix'))


# Binning(cut) --------------------------------------

# Seriesでビニング適用
# --- CategoricalDtypeとして格納
# --- 必要に応じて文字列変換
bins = pd.cut(df.price, 3, labels=['low', 'median', 'high'])
bins.dtype
bins.astype('str')

# DataFrameでビニング適用
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x:pd.cut(x.price, bins=3))


# テーブル表示
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x:pd.cut(x.price, bins=3))\
    .pivot(index='model', columns='price_group', values='price')\
    .style.background_gradient(cmap='Blues')


# Binning(qcut) --------------------------------------

pd.qcut(df.price, q=[0, 0.33, 0.66, 1], labels=['low', 'medium', 'high'])

# テーブル表示
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group=lambda x:pd.qcut(x.price, q=[0, 0.33, 0.66, 1]))\
    .pivot(index='model', columns='price_group', values='price')\
    .style.background_gradient(cmap='Blues')
    
    
# 5.0 GROUPING  --------------------------------------------------------

# 5.1 Aggregations (No Grouping) ------------------------

# 全体に合計演算を適用
# --- 文字列は結合される
df.sum()

# Pandas Seriesに演算適用
# --- 'numpy.int64'で出力される
ans = df.total_price.sum()
type(ans)

# Pandas DataFrame(1列)に演算適用
# --- 'Pandas Series'で出力
# --- DataFrameに櫃王に応じて再変換
ans = df[['total_price']].sum()
type(ans)
ans.to_frame()

# Pandas DataFrame(複数列)に演算適用
# --- 数値列のみを選択
# --- 'Pandas Series'で出力
ans = df\
    .select_dtypes(exclude='object')\
    .drop('order_date', axis=1)\
    .sum()

type(ans)

# 指定列に演算適用
# --- 文字列のSeriesも含めて適用されている
# --- 辞書で系列と適用関数を指定
df.agg([np.sum, np.mean, np.std])
df.agg({'quantity': np.sum, 
        'total_price': [np.sum, np.mean]})

# カウント集計
# --- Seriesに適用
# --- DataFrameに適用（複数列の選択が可能）
df['model'].value_counts()
df[['model', 'category_1']].value_counts()

# 列ごとのユニーク要素数
df.nunique()

# 欠損値カウント
df.isna().sum()

# 単一統計量の計算
# --- 数値列のみ選択して適用される
df.std()

# 複数統計量の計算
# --- 数値列のみ選択して適用される
df.aggregate([np.mean, np.std])


# 5.2 Groupby + Agg ----------------------------------------

# グループ集計
# --- 単一統計量
# --- Pandasが定義する集計メソッドを利用
df.groupby(['city', 'state']).sum()

# グループ集計
# --- 単一統計量
# --- agg()を利用（外部関数を利用）
df.groupby(['city', 'state']).agg(dict(total_price=np.sum))
df.groupby(['city', 'state']).agg({'total_price': np.sum})

# マルチレベルのグループ集計
# --- 複数統計量
df\
    .groupby(['city', 'state'])\
    .agg(dict(total_price=np.sum, 
              quantity=[np.sum, np.mean]))

# 指定列をグループ集計
# --- 予め集計列を選択しておく
# --- 集計関数をリストで与える（集計関数は全列に適用される）
# --- 集計後にインデックスをリセット（インデックスを列に戻す）
summary_df_1 = \
    df[['category_1', 'category_2', 'total_price']]\
        .groupby(['category_1', 'category_2'])\
        .agg([np.sum, np.median])\
        .reset_index()


# 指定列をグループ集計
# --- 集計関数をリストで与える（集計関数は全列に適用される）
# --- 集計後にインデックスをリセット（インデックスを列に戻す）
summary_df_2 = \
    df[['category_1', 'category_2', 'total_price', 'quantity']]\
        .groupby(['category_1', 'category_2'])\
        .agg({'quantity': np.sum, 
            'total_price': np.sum})\
        .reset_index()


# グループごとのNA確認
# --- マルチレベル・インデックス
# --- NAカウント
summary_df_1.columns
summary_df_1.isna().sum()


# 5.3 Groupby + Transform (Apply)----------------------------------

# - Note: Groupby + Assign does not work. No assign method for groups.

# データ準備（日付ごとの合計値）
# --- インデックス設定
# --- グループ化
# --- インデックス/グループごとに合計
summary_df_3 = \
    df[['category_2', 'order_date', 'total_price', 'quantity']]\
        .set_index('order_date')\
        .groupby('category_2')\
        .resample("W")\
        .agg(np.sum)\
        .reset_index()

# データ基準化
# --- データ基準化（x.total_priceと列を指定している）
# --- プロット作成
summary_df_3\
    .set_index('order_date')\
    .groupby('category_2')\
    .apply(lambda x: (x.total_price - x.total_price.mean()) / x.total_price.std())\
    .reset_index()\
    .pivot(index='order_date', columns='category_2', values='total_price')\
    .plot()

# データ基準化
# --- データ基準化（全ての列に適用している）
summary_df_3\
    .set_index(['order_date', 'category_2'])\
    .groupby('category_2')\
    .apply(lambda x: (x - x.mean()) / x.std())\
    .reset_index()


# 5.4 Groupby + Filter

# グループ数の確認
#
summary_df_3.category_2.nunique()
summary_df_3.category_2.value_counts()

# グループごとの末尾データ
summary_df_3\
    .groupby('category_2')\
    .tail(5)

summary_df_3\
    .groupby('category_2')\
    .apply(lambda x: x.iloc[10:20])


# 6.0 RENAMING ------------------------------------------------

# 列名変更
# --- 辞書型で変更を指定
summary_df_2.rename(columns=dict(category_1='Category 1'))
summary_df_2.rename(columns={'total_price': 'Revenue'})

# 規則的な列名変更
# --- ラムダ式と文字列変換の関数を利用
summary_df_2\
    .rename(columns=lambda x: x.replace('_', ' ').title())

# 列名の再定義
summary_df_2.set_axis(['A', 'B', 'C', 'D'], axis=1)

# マルチインデックスの列名変更１
# --- 列名取得（マルチインデックス）
# --- フラット化して列名指定
summary_df_1.columns
summary_df_1.set_axis(['A', 'B', 'C', 'D'], axis=1)

# マルチインデックスの列名変更２
# --- タプルの結合
# --- 内包表記でタプルを結合
# --- 右端の_を削除
"_".join(('total_price', 'median'))
['_'.join(col) for col in summary_df_1.columns.to_list()]
['_'.join(col).rstrip('_') for col in summary_df_1.columns.to_list()]

summary_df_1\
    .set_axis(['_'.join(col).rstrip('_') for col in summary_df_1.columns.to_list()], 
              axis=1)


# 7.0 RESHAPING (MELT & PIVOT_TABLE) ------------------------------------

# データ作成
# --- total_priceをカテゴリ別に集計
# --- Bikeshop / Category 1
bikeshop_revenue_df = \
    df[['bikeshop_name', 'category_1', 'total_price']]\
        .groupby(['bikeshop_name', 'category_1'])\
        .sum()\
        .reset_index()\
        .sort_values('total_price', ascending=False)\
        .rename(columns=lambda x: x.replace('_', ' ').title())

# 確認
bikeshop_revenue_df


# 7.1 Pivot & Melt -----------------------------------------

# Pivot (Pivot Wider)


# ワイド型に変換
# --- pivot()を適用すると列名がマルチインデックスとなる
# --- 列名を変更（単一インデックスに変更）
bikeshop_revenue_wide_df = \
    bikeshop_revenue_df\
        .pivot(index=['Bikeshop Name'], 
            columns=['Category 1'], 
            values=['Total Price'])\
        .reset_index()\
        .set_axis(['Bikeshop Name', 'Mountain', 'Road'], axis=1)
    
# 確認
bikeshop_revenue_wide_df

# プロット作成
bikeshop_revenue_wide_df\
    .sort_values('Mountain')\
    .plot(x='Bikeshop Name', 
          y=['Mountain', 'Road'], 
          kind='barh')


bikeshop_revenue_wide_df\
    .sort_values('Mountain', ascending=False)\
    .style\
    .highlight_max()\
    .format({'Mounttain': lambda x: "$" + str(x)})



from mizani.formatters import dollar_format

usd = dollar_format(prefix='$', digits=0, big_mark=',')

usd([1000])

bikeshop_revenue_wide_df\
    .sort_values('Mountain', ascending=False)\
    .style\
    .highlight_max()\
    .format({'Mountain': lambda x: usd([x])[0], 
             'Road': lambda x: usd([x])[0]})\
    .to_excel("03_pandas_core/bikeshop_revenue_wide.xlsx")


    
# Melt (Pivoting Longer)

bikeshop_revenue_long_df = \
    pd.read_excel("03_pandas_core/bikeshop_revenue_wide.xlsx")\
        .iloc[:, 1:]\
        .melt(value_vars=['Mountain', 'Road'], 
            var_name='Category 1', 
            value_name='Revenue', 
            id_vars='Bikeshop Name')

bikeshop_order = \
    bikeshop_revenue_long_df\
        .groupby('Bikeshop Name')\
        .sum()\
        .sort_values('Revenue')\
        .index\
        .tolist()

from plotnine import (ggplot, aes, geom_col, facet_wrap, 
                      coord_flip, theme_minimal)

bikeshop_revenue_long_df['Bikeshop Name'] = \
    pd.Categorical(bikeshop_revenue_long_df['Bikeshop Name'], 
                   categories=bikeshop_order)
    
    
bikeshop_revenue_long_df.info()

ggplot(
    mapping = aes(x='Bikeshop Name', y='Revenue', fill='Category 1'), 
    data = bikeshop_revenue_long_df) +\
    geom_col() +\
    coord_flip() +\
    facet_wrap('Category 1')


# 7.2 Pivot Table (Pivot + Summarization, Excel Pivot Table)

df\
    .pivot_table(columns=None, 
                 values='total_price', 
                 index='category_1', 
                 aggfunc=np.sum)
    
df\
    .pivot_table(columns='frame_material', 
                 values='total_price', 
                 index='category_1', 
                 aggfunc=np.sum)

df\
    .pivot_table(columns=None, 
                 values='total_price', 
                 index=['category_1', 'frame_material'], 
                 aggfunc=np.sum)

sales_by_cat1_cat2_year_df = df\
    .assign(year=lambda x: x.order_date.dt.year)\
    .pivot_table(
        columns='year', 
        aggfunc=np.sum, 
        index=['category_1', 'category_2'], 
        values=['total_price']
    )



# 7.3 Stack & Unstack --------------------------------------------

# Unstack - Pivots Wider 1 Level (Pivot)

sales_by_cat1_cat2_year_df\
    .unstack(
        level=0, 
        fill_value=0
    )

sales_by_cat1_cat2_year_df\
    .unstack(
        level='category_1', 
        fill_value=0
    )

sales_by_cat1_cat2_year_df\
    .unstack(
        level='category_2', 
        fill_value=0
    )


# Stack - Pivots Longer 1 Level (Melt)

sales_by_cat1_cat2_year_df\
    .stack(level='year')

sales_by_cat1_cat2_year_df\
    .stack(level='year')\
    .unstack(level=['category_1', 'category_2'])


# 8.0 JOINING DATA ----

# データ準備
orderlines_df = pd.read_excel("00_data_raw/orderlines.xlsx")
bike_df = pd.read_excel("00_data_raw/bikes.xlsx")

# Merge (Joining)

pd.merge(
    left=orderlines_df, 
    right=bike_df, 
    left_on="product.id", 
    right_on="bike.id"
    )


# Concatenate (Binding)

# 行の結合
# --- bind_rows
df_1 = df.head(5)
df_2 = df.tail(5)
pd.concat([df_1, df_2], axis=0)

# 列の結合
# --- bind_cols
df_1 = df.iloc[:, :5]
df_2 = df.iloc[:, -5:]
pd.condat([df_1, df_2], axis=1)


# 9.0 SPLITTING (SEPARATING) COLUMNS AND COMBINING (UNITING) COLUMNS

# Separate

# 文字列の分割
# --- 分割したいSeriesを別に操作
df_2 = \
    df['order_date']\
        .astype('str')\
        .str.split('-', expand=True)\
        .set_axis(['year', 'month', 'day'], axis=1)

# 元のデータフレームに結合
pd.concat([df, df_2], axis=1)


# Combine

# 列の結合
df_2['year'] + '-'  + df_2['month'] + '-' + df_2['day']



# 10.0 APPLY 
# - Apply functions across rows 

sales_cat2_daily_df = \
    df[['category_2', 'order_date', 'total_price']]\
        .set_index('order_date')\
        .groupby('category_2')\
        .resample('D')\
        .sum()

# 関数の動作確認
# --- mean: Aggregate（１つの要素を返す）
# --- sqrt: Transform（元のデータと同じ要素数を返す）
np.mean([1, 2, 3])
np.sqrt([1, 2, 3])

# データ集計
# --- Aggregate関数
sales_cat2_daily_df.apply(np.mean)

# データ変換
# --- Transform関数
sales_cat2_daily_df.apply(np.sqrt)

# データ変換
# --- Aggregate関数の結果を列全体に表示
sales_cat2_daily_df.apply(np.mean, result_type='broadcast')
sales_cat2_daily_df.apply(lambda x: np.repeat(np.mean(x), len(x)))

sales_cat2_daily_df\
    .groupby('category_2')\
    .apply(np.mean)

sales_cat2_daily_df\
    .groupby('category_2')\
    .apply(lambda x: np.repeat(np.mean(x), len(x)))

 
# Transform
# --- グループ適用時に使用
sales_cat2_daily_df\
    .groupby('category_2')\
    .transform(np.mean)
 
 
# 11.0 PIPE 
# - Functional programming helper for "data" functions

data = df

# 関数定義
# --- kwargsの動作
def add_column_test(data, **kwargs):
    data_copy = data.copy()
    print(kwargs)
    return None

add_column_test(df, total_price_2=df.total_price * 2)


# 関数定義
# --- 列の追加
def add_column(data, **kwargs):
    data_copy = data.copy()
    data_copy[list(kwargs.keys())] = pd.DataFrame(kwargs)
    return data_copy

add_column(df, total_price_2=df.total_price * 2)


df\
    .pipe(add_column, 
          category_2_lower = df.category_2.str.lower(), 
          category_2_upper = df.category_2.str.upper())
