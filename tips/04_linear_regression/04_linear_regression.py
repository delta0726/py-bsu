# ******************************************************************************
# Title       : Business Science Python Tips
# Theme       : 004 Linear Regression in Python with Scikit Learn
# Created on  : 2021/7/7
# Blog        : https://www.business-science.io/python/2021/07/06/sklearn-linear-regression.html
# Youtube     : https://www.youtube.com/watch?v=A2zlm3NkeDk
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 モデル学習
# 2 結果確認
# 3 予測
# 4 可視化


# 0 準備 ----------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from plotnine import ggplot, aes, geom_point, geom_line
from plotnine.themes import theme_minimal


# データロード
mpg_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
mpg_df

# データ整理
df = mpg_df[['mpg', 'weight']]
df


# 1 モデル学習 --------------------------------------------------------------

# ヘルプ確認
# ?LinearRegression

# データ格納
# --- Y: Pandas Series
# --- X: Pandas DataFrame
y = mpg_df.mpg
X = mpg_df[['weight']]

# モデル学習
model_lr = LinearRegression().fit(X, y)
vars(model_lr)


# 2 結果確認 ---------------------------------------------------------------

# 結果確認
# --- 回帰係数
# --- 切片
model_lr.coef_
model_lr.intercept_


# 3 予測 ------------------------------------------------------------------

# 予測
# --- インサンプル
model_lr.predict(df[['weight']])


# モデル精度
r2_score(y_true = df.mpg,
         y_pred = model_lr.predict(df[['weight']]))


# 4 可視化 ----------------------------------------------------------------

# データ加工
# --- 予測値の追加
df['fitted'] = model_lr.predict(df[['weight']])
df

# プロット作成
ggplot(aes('weight', 'mpg'), df) \
    + geom_point(alpha = 0.5, color = "#2c3e50") \
    + geom_line(aes(y = 'fitted'), color = 'blue') \
    + theme_minimal()
