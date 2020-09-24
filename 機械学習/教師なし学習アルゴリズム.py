import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler # スケール変換器
from sklearn.svm import SVC

# # データ変換の適用
cancer = load_breast_cancer()
# # 訓練セットとテストセットに分ける
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
# print(X_train.shape)
# print(X_test.shape)

# scaler = MinMaxScaler()
# # スケール変換器のfitメソッドにはX_trainのみを与える
# scaler.fit(X_train)

# # データを変換
# X_train_scaled = scaler.transform(X_train)
# # スケール変換の前後のデータ特性をプリント
# # 返還後の配列の形
# print("transformed shape: {}".format(X_train_scaled.shape))
# # 変換前の各特徴量の最小値
# print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
# # 変換前の各特徴量の最大値
# print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
# # 変換後の各特徴量の最小値
# print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
# # 変換後の各特徴量の最大値
# print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

# # テストデータを変換
# X_test_scaled = scaler.transform(X_test)
# # スケール変換の前後のデータ特性をプリント
# # 変換後の各特徴量の最小値
# print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
# # 変換後の各特徴量の最大値
# print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))

#-------------------------------------------------------------------------------------------------
# SVCの学習に対するMimMaxScalerの効果
# 今までやってきたSVCの学習
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuraty:{:.2f}".format(svm.score(X_test, y_test)))

# 0-1スケール変換で前処理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 変換された訓練データで学習
svm.fit(X_train_scaled, y_train)

# 変換されたテストセットでスコア計算
print("Scaled test set accuracy: {:.2f}".format(
    svm.score(X_test_scaled, y_test)))

# 平均を0に分散を1に前処理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 変換された訓練データで学習
svm.fit(X_train_scaled, y_train)
# 変換されたテストセットでスコア計算
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))

