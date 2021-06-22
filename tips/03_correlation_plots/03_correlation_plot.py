# ******************************************************************************
# Title       : Business Science Python Tips
# Tip         : 003
# Theme       : Correlation Plots in Python
# Created on  : 2021/6/23
# Blog        : https://www.business-science.io/python/2021/06/22/plotnine-correlation-plot.html
# Youtube     : https://www.youtube.com/watch?v=3uO8JmjuLUg
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 相関係数行列
# 2 ヒートマップ
# 3 ボーナス: Plotnine


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as p9
import plydata.cat_tools as cat
import matplotlib.pyplot as plt


# データ準備
mpg_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
mpg_df


# 1 相関係数行列 ----------------------------------------------------------------

# 相関係数行列の算出
# --- ピアソン相関係数（デフォルト）
# --- スピアマン相関係数
mpg_df.corr()
mpg_df.corr(method = 'spearman')

# ヘルプ確認
# ?pd.DataFrame.corr


# 2 ヒートマップ --------------------------------------------------------------------

# 相関係数行列の算出
# --- Wide Format
df = mpg_df.corr()
df

# プロット出力
# --- Wide Formatのままインプット
sns.heatmap(data  = df, annot = True)
#plt.show()


# 3 ボーナス: Plotnine --------------------------------------------------------------

# (Long Format)

# 相関係数行列
# --- ロングフォーマット
# --- 表示用に相関係数行列(2桁)の系列を作成
tidy_corr = mpg_df \
    .corr() \
    .melt(ignore_index=False) \
    .reset_index() \
    .set_axis(labels = ["var1", "var2", "value"], axis   = 1) \
    .assign(lab_text = lambda x: np.round(x['value'], 2)) \
    .assign(var1 = lambda x: cat.cat_inorder(x['var1']),
            var2 = lambda x: cat.cat_rev(cat.cat_inorder(x['var2'])))

# プロット作成
p9.ggplot(mapping = p9.aes("var1", "var2", fill = "value"),
          data    = tidy_corr) + \
    p9.geom_tile() + \
    p9.geom_label(
        p9.aes(label = "lab_text"),
        fill = "white",
        size = 8
    ) + \
    p9.scale_fill_distiller() + \
    p9.theme_minimal() + \
    p9.labs(title = "Vehicle Fuel Economy | Correlation Matrix",
            x = "", y = "") + \
    p9.theme(axis_text_x= p9.element_text(rotation=45, hjust = 1),
             figure_size=(8,6))