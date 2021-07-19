# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : Module 7 (Plotnine): Plot Automation
# File        : 03_forecast_plot_automation.py
# Update Date : 2021/7/7
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************



# 0 準備 ---------------------------------------------------------------

# Imports

import pandas as pd
import numpy as np

from plotnine import *
from plotnine import aes
from plotnine import ggplot
from plotnine.facets.facet_wrap import facet_wrap
from plotnine.geoms import geom_line, geom_ribbon
from plotnine.labels import labs
from plotnine.scales.scale_manual import scale_color_manual
from plotnine.scales.scale_xy import scale_x_continuous, scale_x_datetime, scale_y_continuous

from mizani.formatters import dollar_format
from plotnine.themes import theme, theme_minimal
from plotnine.themes.themeable import figure_size, legend_direction, legend_position
from plydata.cat_tools import cat_reorder

from my_pandas_extensions.database import collect_data
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import arima_forecast


# データロード
df = collect_data()

# データ加工
arima_forecast_df = df\
    .summarize_by_time(date_column='order_date', 
                       value_column='total_price', 
                       groups='category_1', 
                       rule='M', 
                       kind='period', 
                       wide_format=True)\
    .arima_forecast(h=12, sp=1)



# 1 プロット作成 ---------------------------------------------------------------

# データ変換
# --- valueとpredictionの列を縦持ちに変換
df_prepped = arima_forecast_df\
    .melt(value_vars=['value', 'prediction'], 
          id_vars=['category_1', 'order_date', 'ci_lo', 'ci_hi'], 
          var_name='variable', 
          value_name='.value')\
    .rename({'.value': 'value'}, axis=1)\
    .assign(order_date=lambda x: x['order_date'].dt.to_timestamp())


# Step 2: Plotting

ggplot(mapping=aes(x='order_date', y='value', color='variable'), 
       data=df_prepped) \
    + geom_ribbon(aes(ymin='ci_lo', ymax='ci_hi'), color=None, alpha=0.2) \
    + geom_line() \
    + facet_wrap('category_1', ncol=1, scales='free_y')\
    + scale_x_datetime(date_labels='%Y', date_breaks='2 year')\
    + scale_y_continuous(labels=dollar_format(big_mark=',', digits=0))\
    + scale_color_manual(values=['red', '#2C3E50'])\
    + theme_minimal()\
    + theme(legend_position='none', 
            subplots_adjust={'wspace': 0.25}, 
            figure_size=(16, 8))\
    + labs(title='Forecast Plot', x='Date', y='Revenue')



# 2 関数化 ---------------------------------------------------------------
# - Make plot_forecast()

# 
data = arima_forecast_df
id_column = 'category_1'
date_column = 'order_date'


# Function Development 

def plot_forecast(data, id_column, date_column, ribbon_alpha=0.2, 
                  facet_ncol=1, facet_scales='free_y', 
                  date_labels='%Y', date_breaks='1 year', wspace=0.25, 
                  figure_size=(16, 8), title='Forcast Plot', 
                  x='Date', y='Revenue'):
    
    arima_forecast_df = data
    required_column = [id_column, date_column, 
                       'value', 'prediction', 'ci_lo', 'ci_hi']
    
    # Data Wrangling
    df_prepped = arima_forecast_df\
        .loc[:, required_column]\
        .melt(value_vars=['value', 'prediction'], 
            id_vars=[id_column, date_column, 'ci_lo', 'ci_hi'], 
            var_name='variable', 
            value_name='.value')\
        .rename({'.value': 'value'}, axis=1)
    
    
    df_prepped[id_column] =cat_reorder(
        c = pd.Categorical(df_prepped[id_column]),
        x = df_prepped['value'], 
        fun=np.mean, 
        ascending=False
    )
    
    # Check for Period
    if df_prepped[date_column].dtype is not 'datetime64[ns]':
        try:
            df_prepped[date_column] = df_prepped[date_column].dt.to_timestamp()
        except:
            try:
                df_prepped[date_column] = pd.to_datetime(df_prepped[date_column])
            except:
                raise Exception("Could not auto-convert `date_column` to datetime64")
    


    # Prepping the Plot
    # --- Geometries
    g = ggplot(mapping=aes(x=date_column, y='value', color='variable'), 
               data=df_prepped) \
        + geom_ribbon(aes(ymin='ci_lo', ymax='ci_hi'), 
                      color=None, alpha=ribbon_alpha) \
        + geom_line() \
        + facet_wrap(id_column, ncol=facet_ncol, scales=facet_scales)
    
    # --- Scales
    g = g \
        + scale_x_datetime(date_labels=date_labels, date_breaks=date_breaks)\
        + scale_y_continuous(labels=dollar_format(big_mark=',', digits=0))\
        + scale_color_manual(values=['red', '#2C3E50'])
    
    # --- Theme and labels
    g = g \
        + theme_minimal()\
        + theme(legend_position='none', 
                subplots_adjust={'wspace': wspace}, 
                figure_size=figure_size)\
        + labs(title=title, x=xlab, y=ylab)

    
    return g
        


# 3 動作確認 ---------------------------------------------------------------

# データ加工
arima_forecast_df = df\
    .summarize_by_time(
        date_column='order_date', 
        value_column='total_price', 
        groups='category_2', 
        rule='M', 
        kind='period', 
        wide_format=True
        )\
    .arima_forecast(
        h=12, 
        sp=1
        )\
    .assign(id_column=lambda x: "revenue")



plot_forecast(data=arima_forecast_df, 
              id_column='category_2', 
              date_column='order_date', 
              facet_ncol=3, 
              facet_scales=None, 
              date_labels='%b %Y', 
              date_breaks='2 year', 
              figure_size=(8, 8), 
              title='Revenue Over Time'
              )
