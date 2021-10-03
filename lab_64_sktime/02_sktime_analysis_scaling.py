# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Marketing Analytics with R & Python
# Chapter     : LAB 64: SKTIME FORECASTING
# Module      : 02_sktime_analysis_scaling.py
# Update Date : 2021/10/03
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 目次
# 1 SCALE


# 0 目次 ---------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np
import plotly.express as px

# データ取得
df = pd.read_csv("data/walmart_item_sales.csv")
df

# データ確認
df.info()


# 1 SCALE -----------------------------------------------------------------------------

# - LL PRO MEMBERS GET THE CODE
# - Tip #2: Handling Errors
# - Tip #3: Review errors & reforecast using different methods

from forecasting.at_scale import run_forecasts
from forecasting.plotting import plot_forecasts

df = pd.read_csv("data/walmart_item_sales.csv")
df

df.info()

# Run Automation

best_forecasts_df = run_forecasts(
    df,
    id_slice=None
)

best_forecasts_df.to_pickle("data/best_forecasts_df.pkl")

best_forecasts_df = pd.read_pickle("data/best_forecasts_df.pkl")

# Plot Automation

plot_forecasts(
    best_forecasts_df,
    facet_ncol=3,
    id_slice=np.arange(0, 12)
)

