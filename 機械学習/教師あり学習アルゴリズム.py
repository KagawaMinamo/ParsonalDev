import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# # データセットの生成
# # forgeデータセットは二つの特徴量を持つ
# X, y = mglearn.datasets.make_forge()
# # データセットのプロット
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # 凡例の表示
# plt.legend(["Class 0", "Class 1"], loc = 4)
# plt.xlabel("First feature") # 第一特徴量
# plt.ylabel("Second feature") # 第二特徴量
# print("X.shape: {}".format(X.shape))

# waveで回帰アルゴリズム
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')# 第3引数はオプションを表しており、今回はmarkerを設定している
plt.ylim(-3, 3)
plt.xlabel("Feature")# 特徴量
plt.ylabel("Target")# 回帰ターゲット

# k-最近傍法によるクラス分類
# 星印で示される3つの新しいデータポイント
# 1-最近傍法での予測では近傍点のラベルが予測されたラベルになる
mglearn.plots.plot_knn_classification(n_neighbors=1)
# 個々のテストする点に対して、近傍点のうちいくつがクラス0に属し、いくつがクラス1に属すか計算し、多いほうのクラスをその点に与える
mglearn.plots.plot_knn_classification(n_neighbors=3)

# 1-最近傍回帰を用いたデータセット予測
mglearn.plots.plot_knn_regression(n_neighbors=1)
# 複数の最近傍点を用いる場合は最近傍点の平均値を用いる
# 3-最近傍回帰を用いたデータセット予測
mglearn.plots.plot_knn_regression(n_neighbors=3)

# 線形モデルによる回帰
# y=w[0]*x[0]+bのように表される
mglearn.plots.plot_linear_regression_wave()


# クラス分類のための線形モデル
# LogistocRegressionを使用したときの決定境界の可視化
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

# LinearSVCを使用したときの決定境界
mglearn.plots.plot_linear_svc_regularization()


