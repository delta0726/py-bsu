# ******************************************************************************
# Title       : DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION
# Theme       : SQL DATABASES (Module 2): Working with SQLAlchemy
# File        : 03_func_collect_data.py
# Update Date : 2021/6/12
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - {sqlalchemy}を使ってSQliteのデータベースを操作する
# - SQLによるデータ取得方法を学ぶ


# ＜目次＞
# 0 準備
# 1 はじめての関数定義
# 2 データ取得関数の作成
# 3 関数の動作確認


# 0 準備 ------------------------------------------------------------------

import sqlalchemy as sql
import pandas as pd


# 1 はじめての関数定義 -------------------------------------------------------

# 関数定義
def my_function(a=1):
    b = 1
    return a + b


# 関数の実行
my_function(a=2)


# 2 データ取得関数の作成 -------------------------------------------------------

def collect_data(conn_string='sqlite:///00_database/bike_orders_database.sqlite'):
    """
    Collect and combines the bike order data

    Args:
        conn_string (str, optional): [description]. 
        Defaults to 'sqlite:///00_database/bike_orders_database.sqlite'.
        
    Returns:
        DataFrame: A pandas data frame that combines data from tables:
            - orderlines: Transactions data
            - bikes: Products data
            - bikeshops: Customers data
    """

    # * 1 Connect Database ----------------------------------
    
    # DB接続
    engine = sql.create_engine(conn_string)
    conn = engine.connect()
    
    # テーブル定義
    # --- engine.table_names()
    table_names = ['bikes', 'bikeshops', 'orderlines']
    
    # データ取得
    # --- 各テーブルを辞書に格納
    data_dict = {}
    for table in table_names:
        data_dict[table] = pd.read_sql(f"SELECT * FROM {table}", con=conn)\
            .drop("index", axis=1)\

    # 確認
    # data_dict.keys()
    # data_dict.items()

    # DB切断
    conn.close()
    
    # 2 Combining & Cleaning Data --------------------------
    
    # データ結合
    joined_df = pd.DataFrame(data_dict['orderlines'])\
        .merge(right=data_dict['bikes'], 
               how='left', 
               left_on='product.id', 
               right_on='bike.id')\
        .merge(right=data_dict['bikeshops'], 
               how='left', 
               left_on='customer.id', 
               right_on='bikeshop.id')

    # * 3 Cleaning -------------------------------------
    
    # データ参照
    df = joined_df
    
    # データ型の変換
    # --- 日付を文字列から日付型に変換
    df['order.date'] = pd.to_datetime(df['order.date'])
    
    # 列の分割
    # --- description列
    # --- ハイフン区切りを複数列に分割
    temp_df = df['description'].str.split(" - ", expand=True)
    df['category.1'] = temp_df[0]
    df['category.2'] = temp_df[1]
    df['frame_material'] = temp_df[2]
       
    # 列の分割
    # --- location列
    # --- ハイフン区切りを複数列に分割
    temp_df = df['location'].str.split(", ", expand=True)
    df['city'] = temp_df[0]
    df['state'] = temp_df[0]
   
    # 列の追加
    # --- total.price
    # --- 総額を計算
    df['total.price'] = df['quantity'] * df['price']

    # 列リストの作成
    # --- df.columns
    cols_to_keep_list = [
        'order.id', 'order.line', 'order.date', 'model', 
        'quantity', 'price', 'total.price', 
        'bikeshop.name', 'category.1', 'category.2', 
        'frame_material', 'city', 'state'
        ]
    
    # 列の選択
    df = df[cols_to_keep_list]
    
    # 列名変更
    # --- ｢.｣を｢_｣に変更
    df.columns = df.columns.str.replace(".", "_")

    # データ確認
    # df.info()

    return df


# 3 関数の動作確認 -------------------------------------------------------

# 動作確認
collect_data()
