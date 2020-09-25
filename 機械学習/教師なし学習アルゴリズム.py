import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler # スケール変換器
from sklearn.svm import SVC
from sklearn.decomposition import PCA

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

#-------------------------------------------------------------------------------------------------
# 腫瘍データのPCAによる可視化
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()

# StandardScalerでスケール変換し、個々の特徴量の分散が1になるようにする
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# データの最初の2つの主成分だけを維持する
pca = PCA(n_components=2)
# cancerデータセットにPCAモデルを適合
pca.fit(X_scaled)
# 最初の主成分に対してデータポイントを変換
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# 第1主成分と第2主成分によるプロット。クラスごとに色分け
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")