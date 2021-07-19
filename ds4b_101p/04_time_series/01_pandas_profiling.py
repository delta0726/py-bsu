# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 4 (Time Series): Profiling Data ----


# 準備 ------------------------------------------------------------

# IMPORTS
import pandas as pd

from pandas_profiling import ProfileReport, profile_report

from my_pandas_extensions.database import collect_data


# データ取得
df = collect_data()
df


# PANDAS PROFILING ------------------------------------------------

# プロファイルの作成
# --- 関数
profile = ProfileReport(df=df)
profile

# プロファイルの作成
# --- メソッド
df.profile_report()


# 大規模データの場合
# --- サンプリングを活用
df.sample(frac=0.5).profile_report()


# ヘルプ確認
# ?pd.DataFrame.profile_report
# ?profile_report()


# Saving Output
df.profile_report().to_file("04_time_series/profile_report.html")


# VSCode Extension - Browser Preview
# --- 右クリック - Open in Browser Preview
# --- ExtensionsのBrowser Previewで表示
