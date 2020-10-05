import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler # スケール変換器
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# # 様々なスケール変換と前処理結果
# mglearn.plots.plot_scaling()

#-------------------------------------------------------------------------------------------------
# # 訓練データとテストデータを同じように変換
# # m合成データを作成
# X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# # 訓練セットとデータセットに分割
# X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# # 訓練セットとテストセットをプロット
# fig, axes = plt.subplots(1, 3, figsize=(13, 4))
# axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
# axes[0].legend(loc='upper left')
# axes[0].set_title("Original Data")

# # MinMaxScalerでデータをスケール変換
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # スケール変換されたデータの特性を可視化
# axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
# axes[1].set_title("Scaled Data")

# # テストセットを訓練セットとは別にスケール変換
# # 最小値と最大値が0,1になる
# # 実際にはやってはいけない
# test_scaler = MinMaxScaler()
# test_scaler.fit(X_test)
# X_test_scaled_badly = test_scaler.transform(X_test)

# # 間違ってスケール変換されたデータを可視化
# axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="training set", s=60)
# axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
# axes[2].set_title("Improperly Scaled Data")

# for ax in axes:
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# fig.tight_layout()

# #-------------------------------------------------------------------------------------------------
# # 主成分分析(PCA)
# mglearn.plots.plot_pca_illustration()
# # 円を書くようにスケール変換
# mglearn.plots.plot_pca_whitening()

# #-------------------------------------------------------------------------------------------------
# # 信号の復元
# # もとの信号源
# S = mglearn.datasets.make_signals()
# plt.figure(figsize=(6, 1))
# plt.plot(S, '-')
# plt.xlabel("Time")
# plt.ylabel("Signal")

# # 混ざった信号を分解して元の成分を取り出す
# # データを混ぜて100次元の状態を作る
# A = np.random.RandomState(0).uniform(size=(100, 3))
# X = np.dot(S, A.T)
# print("Shape of measurements: {}".format(X.shape))

# # NMFを使って3つの信号を復元する
# nmf = NMF(n_components=3, random_state=42)
# S_ = nmf.fit_transform(X)
# print("Recovered signal shape: {}".format(S_.shape))

# # PCA
# pca = PCA(n_components=3)
# H = pca.fit_transform(X)

# # 信号を表示する
# models = [X, S, S_, H]
# names = ['Observations (first three measurements)',
#          'True sources',
#          'NMF recovered signals',
#          'PCA recovered signals']

# fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})
# for model, name, ax in zip(models, names, axes):
#     ax.set_title(name)
#     ax.plot(model[:, :3], '-')

#-------------------------------------------------------------------------------------------------
# # t-SNEを用いた多様体学習
# digits = load_digits()

# fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks': ()})
# for ax, img in zip(axes.ravel(), digits.images):
#     ax.imshow(img)

# # PCAを使ってクラスごとに分離してみる
# # PCAモデルを構築
# pca = PCA(n_components=2)
# pca.fit(digits.data)
# # 数値データを最初の2主成分で変形
# digits_pca = pca.transform(digits.data)
# colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
#           "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
# plt.figure(figsize=(10, 10))
# plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
# plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
# for i in range(len(digits.data)):
#     # 散布図を数字でプロット
#     plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
# plt.xlabel("First principal component")
# plt.ylabel("Second principal component")

# # t-SNEを使ってクラスごとに分離してみる
# tsne = TSNE(random_state=42)
# # fit_transformを使う(transformメソッドがないので)
# digits_tsne = tsne.fit_transform(digits.data)

# plt.figure(figsize=(10, 10))
# plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
# plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
# for i in range(len(digits.data)):
#     # 点ではなく数字をプロット
#     plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
#               color = colors[digits.target[i]],
#               fontdict={'weight': 'bold', 'size': 9})
# plt.xlabel("t-SNE feature 0")
# plt.ylabel("t-SNE feature 1")

#-------------------------------------------------------------------------------------------------
# クラスタリング
# k-meanｓクラスタリング
mglearn.plots.plot_kmeans_algorithm()
# クラスタセンタの境界
mglearn.plots.plot_kmeans_boundaries()

# 合成2次元データを作る
X, y = make_blobs(random_state=1)
# クラスタリングモデルを作る
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# 割り当てられたラベル(3つのくらすたがあるから0~2)
print("Cluster memberships:\n{}".format(kmeans.labels_))
print(kmeans.predict(X))

# データを三角形でプロット
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)

# クラスタセンタの数を変えることもできる
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# クラスタセンタを2つに指定
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])
# クラスタセンタを5つに指定
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

# k-meansがうまくいかない場合
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# ランダムにクラスタデータを作成
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

# 対角線方向に引き伸ばす
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

# データポイントを3つにクラスタリング
kmeans.fit(X)
y_pred = kmeans.predict(X)

# クラスタ割り当てとクラスタセンタをプロット
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
    markers='^', markeredgewidth=2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 合成データセットtwo_moonsデータ作成(ノイズ少な目)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 2つのクラスタにクラスタ分離
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# クラスタ割り当てとクラスタセンタをプロット
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2,
            edgecolor='k')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

