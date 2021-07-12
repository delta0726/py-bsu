# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Marketing Analytics with R & Python
# Chapter     : LAB 59: CUSTOMER LIFETIME VALUE
# Module      : lab_59_customer_ltv.py
# Update Date : 2021/7/11
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜仮想環境＞
# lab_59_customer_ltv_py


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 コホート分析
# 3 機械学習の準備
# 4 機械学習の実行
# 5 作業ファイルの保存
# 6 課題に対するインプリケーション


# 0 準備 ------------------------------------------------------------------------


# ライブラリ
import pandas as pd
import numpy as np
import joblib 

import plydata.cat_tools as cat
import plotnine as pn

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV


# オプション設定
# --- {plotnine}の解像度
pn.options.dpi = 300


# 1 データ準備 -----------------------------------------------------------------

# データロード
cdnow_raw_df = \
    pd.read_csv("data/CDNOW_master.txt",  
                sep   = "\s+", 
                names = ["customer_id", "date", "quantity", "price"])

# データ確認
cdnow_raw_df.info()

# データ加工
# --- dateを数値から日付に変換
cdnow_df = cdnow_raw_df \
    .assign(date = lambda x: x['date'].astype(str)) \
    .assign(date = lambda x: pd.to_datetime(x['date'])) \
    .dropna()

# データ確認
cdnow_df.info()


# 2 コホート分析 ---------------------------------------------------------------

# ＜ポイント＞
# - 「コホート」とは「同じ時期に近しい経験をしている人々のグループ」をいう
# - もともと心理学や社会学において用いられており、
# - 世代や社会的な経験によって被験者を分け、行動や意識にどのような変化が表れるのかを調べる分析手法


# データ取得
# --- 顧客ごとの購入日付
# --- 最初の購入価格を取得
cdnow_first_purchase_tbl = cdnow_df \
    .sort_values(['customer_id', 'date']) \
    .groupby('customer_id') \
    .first()

# データ確認
# --- データフレーム
# --- 最小日付（Pandas Series）
# --- 最大日付（Pandas Series）
cdnow_first_purchase_tbl
cdnow_first_purchase_tbl['date'].min()
cdnow_first_purchase_tbl['date'].max()

# プロット作成
# --- 日付ごとの売上高
# --- 時系列データを元データと異なる頻度で集計することをリサンプリングと呼ぶ
# --- 最初だけ売上高が高く、その後に逓減する
cdnow_df \
    .reset_index() \
    .set_index('date') \
    .filter(['price']) \
    .resample(rule = "MS") \
    .sum() \
    .plot()

# データ取得
# --- カスタマーID
# --- 最初の10IDのみを取得
ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]

# データ集計
# --- 10IDのみ選択して日付ごとの売上高を取得
cdnow_cust_id_subset_df = cdnow_df \
    [cdnow_df['customer_id'].isin(ids_selected)] \
    .groupby(['customer_id', 'date']) \
    .sum() \
    .reset_index()

# プロット作成
# --- ライブラリのみロードしているので略称(pn)を付ける必要がある
pn.ggplot(pn.aes('date', 'price', group = 'customer_id'), 
          data = cdnow_cust_id_subset_df) \
    + pn.geom_line() \
    + pn.geom_point() \
    + pn.facet_wrap('customer_id') \
    + pn.scale_x_date(
        date_breaks = "1 year",
        date_labels = "%Y"
    )


# 3 機械学習の準備 ----------------------------------------------------

#  ＜課題＞
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)


# 3.1 時系列データ分割 ------------------------------

# 期間設定
# --- テストデータの期間
# --- 最終日付
# --- 訓練データとテストデータの境界
n_days   = 90
max_date = cdnow_df['date'].max() 
cutoff   = max_date - pd.to_timedelta(n_days, unit = "d")

# データ分割
# --- 訓練データ
# --- テストデータ
temporal_in_df = cdnow_df[cdnow_df['date'] <= cutoff]
temporal_out_df = cdnow_df[cdnow_df['date'] > cutoff]


# 3.2 特徴量エンジニアリング (RFM) ------------------

# ＜ポイント＞
# - 各種計算を行うことにより、特徴量の列を作成している
#   - Most challenging part
#   - 2-Stage Process
#   - Need to frame the problem
#   - Need to think about what features to include


# アウトサンプルの消費金額 ---------

# データ加工
# --- アウトサンプル
# --- 顧客ごとの消費金額合計
targets_df = temporal_out_df \
    .drop('quantity', axis=1) \
    .groupby('customer_id') \
    .sum() \
    .rename({'price': 'spend_90_total'}, axis = 1) \
    .assign(spend_90_flag = 1)
    

# Recencyの作成 --------------------

# 日付取得
# --- インサンプルの最終日
max_date = temporal_in_df['date'].max()

# データ作成
# --- 最終購入日からインサンプル最終日までの日数
recency_features_df = \
    temporal_in_df \
        [['customer_id', 'date']] \
        .groupby('customer_id') \
        .apply(lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")) \
        .to_frame() \
        .set_axis(["recency"], axis=1)

# データ確認
# --- Recency
recency_features_df


# Frequencyの作成 ------------------

# データ作成
# --- 最終購入日からインサンプル最終日までの日数
frequency_features_df = \
    temporal_in_df \
        [['customer_id', 'date']] \
        .groupby('customer_id') \
        .count() \
        .set_axis(['frequency'], axis=1)

# データ確認
# --- Frequency
frequency_features_df


# 価格特徴量の作成 ----------------

# データ作成
# --- 顧客ごとの合計/平均の消費金額
# --- インサンプルの
price_features_df = \
    temporal_in_df \
        .groupby('customer_id') \
        .aggregate({'price': ["sum", "mean"]}) \
        .set_axis(['price_sum', 'price_mean'], axis = 1)

# データ確認
price_features_df


# 3.3 特徴量の結合 ------------------------------------

# データ結合
# --- 各特徴量をcustomer_idで結合
features_df = \
    pd.concat([recency_features_df, frequency_features_df, price_features_df], 
              axis = 1) \
    .merge(targets_df, 
           left_index  = True, 
           right_index = True, 
           how         = "left") \
    .fillna(0)


# 4 機械学習の実行 ---------------------------------------------------


# 特徴量（共通）
# --- Pandas DataFrame
X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]


# 4.1 NEXT 90-DAY SPEND PREDICTION --------------------

# ラベル
# --- Pandas Series
y_spend = features_df['spend_90_total']

# インスタンス生成
# ---XGBoost（回帰モード）
xgb_reg_spec = \
    XGBRegressor(objective="reg:squarederror", 
                 random_state=123)

# チューニング
# --- クロスバリデーション
# --- 学習率
xgb_reg_model = \
    GridSearchCV(estimator=xgb_reg_spec, 
                 param_grid=dict(learning_rate = [0.01, 0.1, 0.3, 0.5]),
                 scoring = 'neg_mean_absolute_error', 
                 refit   = True, 
                 cv      = 5)

# 学習
xgb_reg_model.fit(X, y_spend)

# 結果確認
vars(xgb_reg_model)

# 結果確認
# --- ベストスコア
# --- 最良の学習率
# --- 最良モデル
xgb_reg_model.best_score_
xgb_reg_model.best_params_
xgb_reg_model.best_estimator_

# 予測
predictions_reg = xgb_reg_model.predict(X)


# 4.2 NEXT 90-DAY SPEND PROBABILITY ------------------

# ラベル
# --- Pandas Series
y_prob = features_df['spend_90_flag']

# インスタンス生成
# ---XGBoost（分類モード）
xgb_clf_spec = \
    XGBClassifier(objective    = "binary:logistic", 
                  random_state = 123)

# チューニング
# --- クロスバリデーション
# --- 学習率
xgb_clf_model = \
    GridSearchCV(estimator=xgb_clf_spec, 
                 param_grid=dict(learning_rate = [0.01, 0.1, 0.3, 0.5]), 
                 scoring = 'roc_auc', 
                 refit   = True, 
                 cv      = 5)

# 学習
xgb_clf_model.fit(X, y_prob)

# 結果確認
vars(xgb_reg_model)

# 結果確認
# --- ベストスコア
# --- 最良の学習率
# --- 最良モデル
xgb_clf_model.best_score_
xgb_clf_model.best_params_
xgb_clf_model.best_estimator_

# 予測
predictions_clf = xgb_clf_model.predict_proba(X)


# 4.3 FEATURE IMPORTANCE (GLOBAL) ---------------------

# 回帰モデル --------------------

# 変数重要度の取得
imp_spend_amount_dict = \
    xgb_reg_model\
        .best_estimator_ \
        .get_booster() \
        .get_score(importance_type = 'gain') 

# データフレームに整理
# --- プロット用にファクターに順序付け
imp_spend_amount_df = \
    pd.DataFrame(data = {
        'feature':list(imp_spend_amount_dict.keys()),
        'value':list(imp_spend_amount_dict.values())
        }) \
    .assign(feature = lambda x: cat.cat_reorder(x['feature'] , x['value']))

# プロット作成
pn.ggplot(pn.aes('feature', 'value'), 
          data = imp_spend_amount_df) \
    + pn.geom_col() \
    + pn.coord_flip()


# 分類モデル --------------------

# 変数重要度の取得
imp_spend_prob_dict = \
    xgb_clf_model\
        .best_estimator_\
        .get_booster()\
        .get_score(importance_type = 'gain') 

# データフレームに整理
# --- プロット用にファクターに順序付け
imp_spend_prob_df = \
    pd.DataFrame(data = {
        'feature':list(imp_spend_prob_dict.keys()),
        'value':list(imp_spend_prob_dict.values())
    }) \
    .assign(feature = lambda x: cat.cat_reorder(x['feature'] , x['value']))

# プロット作成
pn.ggplot(pn.aes('feature', 'value'), 
          data = imp_spend_prob_df) \
    + pn.geom_col() \
    + pn.coord_flip() 


# 5 作業ファイルの保存 ----------------------------------------------------

# 保存用データの作成
predictions_df = pd.concat(
    [pd.DataFrame(predictions_reg).set_axis(['pred_spend'], axis=1), 
     pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'], axis=1), 
     features_df.reset_index()], 
    axis=1)

# 確認
predictions_df

# 保存＆読込
#predictions_df.to_pickle("artifacts/predictions_df.pkl")
#pd.read_pickle('artifacts/predictions_df.pkl')

# Save Importance
#imp_spend_amount_df.to_pickle("artifacts/imp_spend_amount_df.pkl")
#imp_spend_prob_df.to_pickle("artifacts/imp_spend_prob_df.pkl")

#pd.read_pickle("artifacts/imp_spend_amount_df.pkl")

# モデル保存
#joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
#joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')

# モデル読込
#model = joblib.load('artifacts/xgb_reg_model.pkl')
#model.predict(X)


# 6 課題に対するインプリケーション ---- 

# 6.1 Which customers have the highest spend probability in next 90-days? 
#     - Target for new products similar to what they have purchased in the past

predictions_df \
    .sort_values('pred_prob', ascending=False)

# 6.2 Which customers have recently purchased but are unlikely to buy? 
#    - Incentivize actions to increase probability
#    - Provide discounts, encourage referring a friend, nurture by letting them know what's coming

predictions_df \
    [
        predictions_df['recency'] > -90
    ] \
    [
        predictions_df['pred_prob'] < 0.20
    ] \
    .sort_values('pred_prob', ascending=False)


# 6.3 Missed opportunities: Big spenders that could be unlocked ----
#    - Send bundle offers encouraging volume purchases
#    - Focus on missed opportunities

predictions_df \
    [
        predictions_df['spend_90_total'] == 0.0
    ] \
    .sort_values('pred_spend', ascending=False) 
