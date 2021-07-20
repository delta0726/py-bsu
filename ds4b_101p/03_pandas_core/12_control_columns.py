# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : 列の選択/削除
# File        : 12_select_column.py
# Update Date : 2021/6/16
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 列名を指定した抽出（ブラケット）
# 2 列名を指定した抽出（loc）
# 3 列名を指定した抽出（filter）
# 4 列番号で列抽出（iloc）
# 5 正規表現による列抽出
# 6 キーワードによる列抽出
# 7 データ型による列抽出
# 8 列の削除
# 9 列の並び替え
# 10 特定列を先頭にする
# 11 複数の特定列を先頭する
# 12 列名の変更


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import pandas as pd


# データロード
df = pd.read_csv("00_data_raw/bike_agg_data.csv")
df


# 1 列名を指定した抽出（ブラケット）-------------------------------------------

# ＜ポイント＞
# - 最も基本的な抽出方法
# - メソッドチェーンの最初では使いやすいが、途中では使いにくい


# 列名を指定して選択
# --- 直接指定
df[['order_date', 'order_id', 'order_line']]

# 列名を指定して選択
# --- 変数で指定
select_cols = ['order_date', 'order_id', 'order_line']
df[select_cols]


# 2 列名を指定した抽出（loc）-------------------------------------------

# ＜ポイント＞
# - locプロパティを使って列選択を行う
# - Pythonでは列選択はlocプロパティと思っている人が多い（filterよりlocを使用するほうが良い）


# 列名を指定して選択
# --- 直接指定
df.loc[:, ['order_date', 'order_id', 'order_line']]

# 全ての列を抽出
df.loc[:, :]


# 3 列名を指定した抽出（filter）-------------------------------------------

# ＜ポイント＞
# - SQLだと｢filter｣は行抽出を意味するが、Pandasでは列選択の意味を持つ


df.filter(['order_date', 'order_id', 'order_line'])


# 4 列番号で列抽出（iloc） ----------------------------------------------

# 複数列の抽出
# --- 列番号を指定して抽出
# --- データフレーム全体を抽出
# --- 最後の3列を抽出
df.iloc[:, 0:3]
df.iloc[:, :]
df.iloc[:, -3:]


# 5 正規表現による列抽出 ----------------------------------------------------

# 正規表現で列抽出
# --- filter()は主に正規表現での抽出に使用する
# --- ^model: starts_with("model")
# --- price$: ends_with("price")
df.filter(regex="(^model)|(^cat)", axis=1)
df.filter(regex="(price$)|(date$)", axis=1)


# 6 キーワードによる列抽出 -------------------------------------------------------

# 文字列を選択して列抽出
# --- 列名を抽出して文字列を判定
df.loc[:, lambda x: df.columns.str.startswith(("model", "cat"))]
df.loc[:, lambda x: df.columns.str.endswith(("price", "date"))]


# 7 データ型による列抽出 --------------------------------------------------------

# 文字列の抽出
df.select_dtypes(include='object').info()


# 8 列の削除 ------------------------------------------------------------------

# 列の削除
df\
    .loc[:, ['model', 'category_1', 'category_2']]\
    .drop(['model'], axis=1)

# データ型による削除
df.select_dtypes(exclude=object).info()
df.select_dtypes(exclude='int64').info()
df.select_dtypes(exclude='datetime64').info()


# 9 列の並び替え ----------------------------------------------------------------

# ＜ポイント＞
# - 基本的に列名指定で並び替えすることが多い
# - 列名指定と先頭名指定を組み合わせることも多い


# 列名の取得
# --- コンソール上のリストをコピー
df.columns

# ＜参考：出力例＞
# ['order_id', 'order_line', 'order_date', 'quantity', 'price',
#        'total_price', 'model', 'category_1', 'category_2', 'frame_material',
#        'bikeshop_name', 'city', 'state'],

# 必要な列を任意の順序で指定
df[['order_id', 'order_date', 'model']]

# 同じ列を複数用いることも可能
df[['order_id', 'order_date', 'model', 'model']]

# 先頭名と組み合わせて並び替え


# 10 特定列を先頭にする ----------------------------------------------------------

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


# 11 複数の特定列を先頭する ----------------------------------------------------

# 単純な方法
col_lst = df.columns.tolist()
col_lst.remove('model')
col_lst.remove('category_1')
col_lst.remove('category_2')
df[['model', 'category_1', 'category_2', *col_lst]]

# 内包表記の活用
col_lst = df.columns.tolist()
cols_to_front = ['model', 'category_1', 'category_2']
l2 = [col for col in col_lst if col not in cols_to_front]
df[[*cols_to_front, *l2]]

# concatを活用
df1 = df[['model', 'category_1', 'category_2']]
df2 = df.drop(['model', 'category_1', 'category_2'], axis=1)
pd.concat([df1, df2], axis=1)
df.info()

# データ型に着目
# --- 先頭行にするのが文字列(object)であることに着目
df1 = df.select_dtypes(include='object')
df2 = df.select_dtypes(exclude='object')
pd.concat([df1, df2], axis=1)
df.info()


# 12 列名の変更--------------------------------------------------------

# 単一列の列名変更
df[['category_1']].rename(columns=dict(category_1='Category_1'))
df[['category_1']].rename(columns={'category_1': 'Category 1'})

# 複数列の列名変更
df[['category_1', 'category_2']]\
    .rename(columns=dict(category_1='Category_1',
                         category_2='Category_2'))

df[['category_1', 'category_2']]\
    .rename(columns={'category_1': 'Category_1',
                     'category_2': 'Category_2'})

# 列名の再定義
# --- 元の列名を指定する必要がない
# --- 列数と同じ要素数のリストが必要
df[['category_1', 'category_2']].set_axis(['cat1', 'cat2'], axis=1)

# 規則的な列名変更
# --- ラムダ式と文字列変換の関数を利用
df.rename(columns=lambda x: x.replace('_', ' ').title())
