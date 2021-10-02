# ******************************************************************************
# Title       : Business Science Python Tips
# Theme       : 002 Siuba: Dplyr for Python
# Created on  : 2021/6/19
# Blog        : https://www.business-science.io/python/2021/06/08/siuba.html
# Youtube     : https://www.youtube.com/watch?v=ySGCedmeQ0s
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 SIUBA: GROUP BY + SUMMARIZE
# 2 SIUBA: GROUP BY + MUTATE
# 3 データフレームのスタイル表示(PANDAS)


# 0 準備 -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

from siuba import _

# 参考
# --- 関数インポートがなぜかワークしない
# --- 代替策
# from siuba.dply.verbs import group_by, mutate, select, summarize, group, ungroup
# from siuba.dply.verbs import *


# データ準備
mpg_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
mpg_df


# 1 SIUBA: GROUP BY + SUMMARIZE --------------------------------------------------------

# グループ集計
# --- ungroup()をしなくてもPandasDataFrameオブジェクトが出力される
weight_by_cyl_df = \
    mpg_df >> \
        group_by("cylinders") >> \
        summarize(
            mean_weight = np.mean(_.weight),
            sd_weight   = np.std(_.weight)
        )

weight_by_cyl_df
type(weight_by_cyl_df)


# 2 SIUBA: GROUP BY + MUTATE ------------------------------------------------------------

# 集計列の追加
# --- mutate()の時点ではPandasGroupByオブジェクト
# --- ungroup()を実行するとPandasDataFrameオブジェクトが出力される
mpg_demeaned_by_cyl_df = \
    mpg_df >> \
        select('name', 'cylinders', 'mpg') >> \
        group_by("cylinders") >> \
        mutate(mean_mpg = np.mean(_.mpg)) >> \
        ungroup() >> \
        mutate(mpg_demeaned_by_cyl = _.mpg - _.mean_mpg)

mpg_demeaned_by_cyl_df
type(mpg_demeaned_by_cyl_df)


# 3 データフレームのスタイル表示 -------------------------------------------------------------

mpg_demeaned_by_cyl_df[['name', 'cylinders', 'mpg_demeaned_by_cyl']] \
    .sort_values('mpg_demeaned_by_cyl', ascending = False) \
    .style \
    .background_gradient()
