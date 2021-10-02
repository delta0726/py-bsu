# ******************************************************************************
# Title       : Business Science Python Tips
# Theme       : 006 Reading Many CSV Files
# Created on  : 2021/10/1
# Blog        : https://www.business-science.io/python/2021/09/21/python-read-csv.html
# Youtube     : https://www.youtube.com/watch?v=TN_Cvyq_rxE
# Github      : https://github.com/business-science/free_python_tips
# ******************************************************************************


# ＜目次＞
# 0 準備
# 方法1：FORループによる読込
# 方法2：map関数を使う方法
# 方法3： リスト内包表記を使う方法


# 0 準備 ----------------------------------------------------------------------

# ライブラリ
import os
import sys

import pandas as pd
import glob


# パス一覧の取得
path = "tips/06_read_many_csv/car_data/"
all_files = glob.glob(path + "*.csv")
all_files


# 方法1：FORループによる読込 -----------------------------------------------------

# ＜ポイント＞
# - FORループでリストにデータフレームを格納して結合
# - 多くのユーザーがこの方法を行うが高速処理には向かない

# 空のリスト
li = []

# CSV読込み
# --- ファイルの1列目がパスになっている
# --- デバッグ用：filename = all_files[0]
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

# データ確認
# --- 1つめのデータフレーム
li[0]

# データフレーム結合
df = pd.concat(li, axis=0, ignore_index=True)
df


# 方法2：map関数を使う方法 -------------------------------------------------------

# ＜ポイント＞
# - Forループを排除することで簡素に記述することが可能
#   --- 格納用のリストをあらかじめ用意する必要もない

# CSV読込
# --- map関数によりイテレータを作成
li_mapper = map(lambda x: pd.read_csv(x, index_col=None, header=0), all_files)

# リストに出力
# --- イテレータのままではデータが確認できない
li_2 = list(li_mapper)

# データフレーム結合
df = pd.concat(li_2, axis=0, ignore_index=True)
df

# 確認：イテレータの中身
for i in li_mapper:
    print(i)


# 方法3： リスト内包表記を使う方法 -----------------------------------------------

# ＜ポイント＞
# - Forループを排除することで簡素に記述することが可能
#   --- 格納用のリストをあらかじめ用意する必要もない
#   --- 本稿では最も簡素な記法と考えている

# CSV読込
# --- 読込ながら直接リストに格納
li_3 = [pd.read_csv(filename, index_col=None, header=0) for filename in all_files]

# データフレーム結合
df = pd.concat(li_3, axis=0, ignore_index=True)
df
