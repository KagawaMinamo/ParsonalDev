import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler # スケール変換器

# データ変換の適用
cancer = load_breast_cancer()
# 訓練セットとテストセットに分ける
X_train, X_test, y_trsin, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
# スケール変換器のfitメソッドにはX_trainのみを与える
scaler.fit(X_train)

# データを変換
X_train_scaled = scaler.transform(X_train)
# スケール変換の前後のデータ特性をプリント
# 返還後の配列の形
print("transformed shape: {}".format(X_train_scaled.shape))
# 変換前の各特徴量の最小値
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
# 変換前の各特徴量の最大値
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
# 変換後の各特徴量の最小値
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
# 変換後の各特徴量の最大値
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

# テストデータを変換
X_test_scaled = scaler.transform(X_test)
# スケール変換の前後のデータ特性をプリント
# 変換後の各特徴量の最小値
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
# 変換後の各特徴量の最大値
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
