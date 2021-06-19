# ******************************************************************************
# Title       : Business Science Python Tips
# Tip         : 001
# Theme       : Pandas Profiling
# Created on  : 2021/6/17
# Blog        : https://www.business-science.io/python/2021/06/01/pandas-profiling.html
# Youtube     : https://www.youtube.com/watch?v=-Cdv9C9hLeE
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************



# LIBRARIES ----

import os
import pandas as pd
import pandas_profiling as pf

from plotnine import (
    ggplot, aes, geom_point, geom_smooth, labs,
    theme_xkcd
)


os.getcwd()

# DATASET ----

mpg_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")


# PANDAS PROFILING ----

rpt = pf.ProfileReport(mpg_df)

rpt.to_file("tips/01_pandas_profiling/profile_report.html")


# PLOTNINE BONUS ----

ggplot(
    aes('horsepower', 'mpg'),
    data = mpg_df
) \
    + geom_point() \
    + geom_smooth(
        method = 'loess', 
        span   = 0.8,
        color  = "dodgerblue"
    ) \
    + labs(
        title = "Trend of Horsepower vs Fuel Economy"
    ) \
    + theme_xkcd()

