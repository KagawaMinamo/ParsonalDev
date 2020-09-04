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
from sklearn.linear_model import Ridge 

# 腫瘍データをロードする
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
# データの中身を見る
print("Shape of cancer data:", cancer.data.shape)
# クラスごとのサンプルの個数
print("Sample counts per class:\n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
# 個々の特徴量の意味を示す
print("Feature names:\n{}".format(cancer.feature_names))

# 実世界の回帰データセットとしてboston_housingを用いる
# 1970年代のボストン近郊の住宅地の住宅価格の中央値を犯罪率、チャールズ川からの距離、高速道路への利便性などから予測するもの
boston = load_boston()
# 506のデータポイントと13の特徴量があることがわかる
print("Data shape: {}".format(boston.data.shape))
# 特徴量の積を見る
# 104は、13C2=78 78+13=91　を足したもの(13の特徴量から2つ重複ありで選んでる)
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

#--------------------------------------------------------------------------------------------------------------------
# データを訓練セットとテストセットに分割
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
# インスタンスの生成
clf = KNeighborsClassifier(n_neighbors=3)
# 訓練セットを用いてクラス分類器を訓練(データセット保存)
clf.fit(X_train, y_train)
# テストデータに対して予測
print("Test set prediction: {}".format(clf.predict(X_test)))
# テストの評価
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# K-近傍法の解析
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
#kが1,3,9の時のクラス0の割り当てる場合と、クラス1に割り当てる場合の決定境界を描画
for n_neighbors, ax in zip([1, 3, 9], axes):
    #fitメソッドは自分自身を返すので、1行でインスタンスを生成してfitすることができる
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

#------------------------------------------------------------------------------------------------------------------------
# # モデルの複雑さと汎化性能の関係を確認
# # データセットを訓練セットとテストセットに分割
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# training_accuracy = []
# test_accuracy = []
# # neighborsを1から10まで試す
# neighbors_settings = range(1, 11)

# # 訓練セットに対する精度とテストセットに対する性能を近傍点の数に対して評価
# for n_neighbors in neighbors_settings:
#     # モデルを構築
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     # 訓練セット制度を記録
#     training_accuracy.append(clf.score(X_train, y_train))
#     # 汎化制度を記録
#     test_accuracy.append(clf.score(X_test, y_test))
    
# # 結果を図に表示
# plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()

#--------------------------------------------------------------------------------------------------------------------------

X, y = mglearn.datasets.make_wave(n_samples=40)
# waveデータセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# モデルのインスタンス作成
reg = KNeighborsRegressor(n_neighbors=3)
# 訓練データと訓練ターゲットを用いてモデルを学習させる
reg.fit(X_train, y_train)
# テストセットに対して予測
print("Test set predictions:\n{}".format(reg.predict(X_test)))
# モデルの評価
# R^2スコアは決定係数とも呼ばれ、回帰モデルの予測の正確さを測る指数で0から1までの値を取る
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

#--------------------------------------------------------------------------------------------------------------------------
# 1次元のデータセットに対して、すべての値に対する予測値がどのようになるか確認
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# データポイント作成
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1,3,9の近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train, y_train),reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")

#--------------------------------------------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------------------------------------------------
# リッジ回帰
# 線形回帰の時より訓練セットに対するスコアが低い
ridge = Ridge().fit(X_train, y_train)
print("Trainig set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# alphaを増やすと訓練セットに対する性能は低下するが、汎化にはよくなる
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))