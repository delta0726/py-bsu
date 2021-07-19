# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 5 (Programming): Functions ----


# 0 準備 -------------------------------------------------------------

# Imports

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data

# データロード
df = collect_data()

# 1.0 EXAMINING FUNCTIONS ----

# Pandas Series Function
# ?pd.Series.max
# ?np.max

# データ型の確認
# Pandas Series
type(df.total_price)

# メソッドの適用
# --- 最大値の取得
df.total_price.max()
pd.Series.max("a")

# 変数に関数を格納
# --- 関数もオブジェクトなので変数に入れることが可能
# --- 新しい関数を定義しているイメージ
my_max = pd.Series.max
my_max(df.total_price)


# Pandas Data Frame Function
# ?pd.DataFrame.aggregate


# データ集計
# --- 各列を合計（文字列の場合は結合）
# --- メソッドとして適用（第1引数のselfにはdfが自動的に適用されている）
# --- selfに明示的にdfを参照
df.aggregate(func=np.sum)
pd.DataFrame.aggregate(self=df, func=np.sum)


# データ集計
# --- total_priceの分位点を出力(q=0.5は中央値)
# --- **kwargsを使ってnp.quantile()の引数を渡している
pd.DataFrame.aggregate(self=df[['total_price']], func=np.quantile, q=0.5)


# 2.0 OUTLIER DETECTION FUNCTION ------------------------------------
# - Works with a Pandas Series

# データ準備
x = df['total_price']


def detect_outliers(x, iqr_multiplier=1.5, how="both"):
    # IQR Logic
    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)
    iqr = q75 - q25
    
    # Define Limit
    lower_limit = q25 - 1.5 * iqr
    upper_limit = q75 + 1.5 * iqr
    
    # Check Outlier
    outliers_upper = x >= upper_limit
    outliers_lower = x <= lower_limit
    
    if how == "both":
        outliers = outliers_upper | outliers_lower
    elif how == "lower":
        outliers = outliers_lower
    else:
        outliers = outliers_upper
    
    # Output
    return outliers


detect_outliers(df['total_price'], iqr_multiplier=0.5)
df[detect_outliers(df['total_price'], iqr_multiplier=0.5)]
detect_outliers(1)


# 3.0 EXTENDING A CLASS ----

