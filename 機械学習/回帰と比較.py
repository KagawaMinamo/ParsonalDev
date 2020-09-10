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
import os
from sklearn.tree import  DecisionTreeRegressor # 決定木
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from mpl_toolkits.mplot3d import Axes3D, axes3d


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
# # Lasso
# # 係数が0になるように制約を書けるがリッジ回帰と違ってL1正規化と呼ばれる
# lasso = Lasso().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
# print("Number of features used:", np.sum(lasso.coef_ != 0))

# # max_iterの値を増やしている
# # 酢やしておかないとモデルがmax_iterを増やせと警告する
# lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
# print("Number of features used:", np.sum(lasso001.coef_ != 0))

# # alphaを小さくしすぎるとリッジ回帰の場合同様、正規化の効果が薄れ過剰適合が発生
# lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
# print("Number of features used:", np.sum(lasso00001.coef_ != 0))

# # 係数をプロット
# plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
# plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
# plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
# plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
# plt.legend(ncol=2, loc=(0, 1.05))
# plt.ylim(-25, 25)
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")

#----------------------------------------------------------------------------------------------------------------------------
# # 線形モデルによる多クラス分類
# # 単純な3クラス分類データセットに対して1対その他手法を適応させてみる
# # 1対その他は2クラス分類アルゴリズムを多クラス分類アルゴリズムに拡張すること
# X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(["Class 0", "Class 1", "Class 2"])

# # LineaeSVCクラス分類器をデータセットに学習させてみる
# linear_svm = LinearSVC().fit(X, y)
# print("Coefficient shape: ", linear_svm.coef_.shape)
# print("Intercept shape: ", linear_svm.intercept_.shape)

# # 3つのクラス分類器による直線の可視化
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))

# # 1対その他クラス分類器による多クラス分類の決定境界
# mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')

#----------------------------------------------------------------------------------------------------------------------------
# 本編とは関係ない
# # メソッドチェーン
# # 1行でモデルのインスタンスを生成して訓練する
# logreg = LogisticRegression().fit(X_train, y_train)
# # scikit-learnでは、fitとpredictに対してメソッドチェーンがよく使われる
# logreg = LogisticRegression()
# y_pred = logreg.fit(X_train, y_train).predict(X_test)

#----------------------------------------------------------------------------------------------------------------------------
# ナイーブベイズクラス分類器
# 線形モデルによく似たクラス分類器
# 訓練が線形モデルよりも高速だが、汎化性能はLogisticRegressionやLinearSVCよりわずかに劣る
# それぞれ4つの2値特徴量を持つ4つのデータポイントがある、クラス0と1がある
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    # クラスに対してループ
    # それぞれの特徴量ごとに非ゼロの数を数える
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts: \n{}".format(counts))

#----------------------------------------------------------------------------------------------------------------------------
# # 決定木
# # 計算機のメモリ(RAM)価格の履歴データセットをプロットしてみる
# ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel("Year")
# plt.ylabel("Price in $/Mbyte")

# # 2000年目でのデータを使ってそれ以降を予測してみる
# # 過去データを用いて2000年以降の価格を予測する
# data_train = ram_prices[ram_prices.date < 2000]
# data_test = ram_prices[ram_prices.date >= 2000]

# # 日付に基づいて予測
# X_train = data_train.date[:, np.newaxis]
# # データとターゲットの関係を単純にするために対数変換
# y_train = np.log(data_train.price)

# tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
# linear_reg = LinearRegression().fit(X_train, y_train)

# # すべての価格を予測
# X_all = ram_prices.date[:, np.newaxis]

# pred_tree = tree.predict(X_all)
# pred_lr = linear_reg.predict(X_all)

# # 対数変換をキャンセルするために逆変換
# price_tree = np.exp(pred_tree)
# price_lr = np.exp(pred_lr)

# # 決定木モデルと線形モデルの予測結果と実際のデータを比較
# plt.semilogy(data_train.date, data_train.price, label="Training data")
# plt.semilogy(data_test.date, data_test.price, label="Test data")
# plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
# plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
# plt.legend()

#----------------------------------------------------------------------------------------------------------------------------
# # 決定木のアンサンブル法
# # アンサンブル法とは、複数の機械学習モデルを組み合わせることでより強力なモデルを構築する手法
# # 様々なデータセットに対するクラス分類や回帰に対して有効

# # ランダムフォレスト
# # 少しずつ異なる決定木をたくさん集めたもの
# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42)

# forest = RandomForestClassifier(n_estimators=5, random_state=2)
# forest.fit(X_train, y_train)

# # それぞれの決定木で学習された決定境界とランダムフォレストによって行われる集合的な予測
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     ax.set_title("Tree {}".format(i))
#     mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],alpha=.4)
# axes[-1, -1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

#----------------------------------------------------------------------------------------------------------------------------
# 線形モデルと非線形特徴量
# 線形分離が不可能な2クラス分類データセット
X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 特徴量を拡張して3次元にしてみる
# 2番目の特徴量の2乗を追加
X_new = np.hstack([X, X[:, 1:] ** 2])

figure = plt.figure()
# 3Dで可視化
ax = Axes3D(figure, elev=-152, azim=-26)
# y==0をプロットしてからy==1の点をプロット
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 線形決定境界の描画
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')

ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
