# ******************************************************************************
# Title       : Business Science Python Tips
# Tip         : 002
# Theme       : Siuba: Dplyr for Python
# Created on  : 2021/6/19
# Blog        : https://www.business-science.io/python/2021/06/08/siuba.html
# Youtube     : https://www.youtube.com/watch?v=ySGCedmeQ0s
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************


# LEARNING PANDAS ----
# - Siuba is great for when you are coming from R to Python (like me)
# - Teams use Pandas: 99% of data wranging code is written with Pandas
# - Better Learn Pandas if you want to be part of the Team


# LIBRARIES ----
import numpy as np
import pandas as pd

from siuba import _
from siuba.dply.verbs import group_by, mutate, select, summarize, ungroup

# DATASET ----

mpg_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
mpg_df


# 1.0 GROUP BY + SUMMARIZE ---------------------------------------------------------

# Goal: Mean and Standard Deviation of weight by engine size

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


# 2.0 GROUP BY + MUTATE ------------------------------------------------------------
# Goal: De-mean the mpg by average of each cylinder

# 集計列の追加
# --- mutate()の時点ではPndasGroupByオブジェクト
# --- ungroup()を実行するとPandasDataFrameオブジェクトが出力される
mpg_demeaned_by_cyl_df = \
    mpg_df >> \
        select('name', 'cylinders', 'mpg') >> \
        group_by("cylinders") >> \
        mutate(
            mean_mpg = np.mean(_.mpg)
        ) >> \
        ungroup() >> \
        mutate(
            mpg_demeaned_by_cyl = _.mpg - _.mean_mpg
        )

mpg_demeaned_by_cyl_df
type(mpg_demeaned_by_cyl_df)


# 3.0 PANDAS ----------------------------------------------------------------------

mpg_demeaned_by_cyl_df[['name', 'cylinders', 'mpg_demeaned_by_cyl']] \
    .sort_values('mpg_demeaned_by_cyl', ascending = False) \
    .style \
    .background_gradient()
