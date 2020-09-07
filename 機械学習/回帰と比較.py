import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor# k-近傍回帰 
from sklearn.linear_model import LinearRegression# 線形回帰
from sklearn.linear_model import Ridge # リッジ回帰
from sklearn.linear_model import Lasso #ラッソ
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs


# # データを訓練セットとテストセットに分割
# X, y = mglearn.datasets.make_forge()
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
# # インスタンスの生成
# clf = KNeighborsClassifier(n_neighbors=3)
# # 訓練セットを用いてクラス分類器を訓練(データセット保存)
# clf.fit(X_train, y_train)
# # テストデータに対して予測
# print("Test set prediction: {}".format(clf.predict(X_test)))
# # テストの評価
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# # K-近傍法の解析
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# #kが1,3,9の時のクラス0の割り当てる場合と、クラス1に割り当てる場合の決定境界を描画
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     #fitメソッドは自分自身を返すので、1行でインスタンスを生成してfitすることができる
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{} neighbor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)

# X, y = mglearn.datasets.make_wave(n_samples=40)
# # waveデータセットを訓練セットとテストセットに分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # モデルのインスタンス作成
# reg = KNeighborsRegressor(n_neighbors=3)
# # 訓練データと訓練ターゲットを用いてモデルを学習させる
# reg.fit(X_train, y_train)
# # テストセットに対して予測
# print("Test set predictions:\n{}".format(reg.predict(X_test)))
# # モデルの評価
# # R^2スコアは決定係数とも呼ばれ、回帰モデルの予測の正確さを測る指数で0から1までの値を取る
# print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

# #--------------------------------------------------------------------------------------------------------------------------
# # 1次元のデータセットに対して、すべての値に対する予測値がどのようになるか確認
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# # データポイント作成
# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     # 1,3,9の近傍点で予測
#     reg = KNeighborsRegressor(n_neighbors=n_neighbors)
#     reg.fit(X_train, y_train)
#     ax.plot(line, reg.predict(line))
#     ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
#     ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

#     ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train, y_train),reg.score(X_test, y_test)))
#     ax.set_xlabel("Feature")
#     ax.set_ylabel("Target")
# axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")

# #--------------------------------------------------------------------------------------------------------------------------
# 線形回帰(通常最小二乗法)
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
# coef_には係数、intercept_には切片が格納される
print("lr.coef_:", lr.coef_)
print("lr.intercept_:", lr.intercept_)

# 訓練セットとテストセットに対する性能
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

# 訓練セットとテストセットで性能が大きく異なると、過剰適合が起こっている兆候
# 複雑度を制御できるものを探さなければならない
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# #----------------------------------------------------------------------------------------------------------------------------
# リッジ回帰
# 線形回帰の時より訓練セットに対するスコアが低い
ridge = Ridge().fit(X_train, y_train)
print("Trainig set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# alphaを増やすと訓練セットに対する性能は低下するが、汎化にはよくなる
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# alphaを小さくすると係数の制約が小さくなる
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# # 係数をプロット
# plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

# plt.plot(lr.coef_, 'o', label="LinearRegression")
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# xlims = plt.xlim()
# plt.hlines(0, xlims[0], xlims[1])
# plt.xlim(xlims)
# plt.ylim(-25, 25)
# plt.legend()

# # 学習曲線
# mglearn.plots.plot_ridge_n_samples()

#----------------------------------------------------------------------------------------------------------------------------
# Lasso
# 係数が0になるように制約を書けるがリッジ回帰と違ってL1正規化と呼ばれる
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))

# max_iterの値を増やしている
# 酢やしておかないとモデルがmax_iterを増やせと警告する
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso001.coef_ != 0))

# alphaを小さくしすぎるとリッジ回帰の場合同様、正規化の効果が薄れ過剰適合が発生
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso00001.coef_ != 0))

# 係数をプロット
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

#----------------------------------------------------------------------------------------------------------------------------
