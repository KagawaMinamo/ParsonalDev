import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor# k-近傍回帰 
from sklearn.linear_model import LinearRegression# 線形回帰
from sklearn.linear_model import Ridge # リッジ回帰
from sklearn.linear_model import Lasso # ラッソ
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

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

#----------------------------------------------------------------------------------------------------------------------------
# LogisticRegressionを使用して詳しく解析
# デフォルトはC=1
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
# 訓練セットとテストセットの精度がとても近いと適合不足の可能性が高い
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# 柔軟なモデルにする
# Cを増やす
# 訓練セットもテストセットも精度が向上した
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# Cを減らす
# 訓練セットもテストセットもデフォルトより精度が悪くなる
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# # 正規化パラメータCに対して学習された係数を見る
# plt.plot(logreg.coef_.T, 'o', label="C=1")
# plt.plot(logreg100.coef_.T, '^', label="C=100")
# plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# xlims = plt.xlim()
# plt.hlines(0, xlims[0], xlims[1])
# plt.xlim(xlims)
# plt.ylim(-5, 5)
# plt.xlabel("Feature")
# plt.ylabel("Coefficient magnitude")
# plt.legend()

# # 分類制度プロット
# for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
#     lr_l1 = LogisticRegression(C=C, solver='liblinear', penalty="l1").fit(X_train, y_train)
#     print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
#           C, lr_l1.score(X_train, y_train)))
#     print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
#           C, lr_l1.score(X_test, y_test)))
#     plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# xlims = plt.xlim()
# plt.hlines(0, xlims[0], xlims[1])
# plt.xlim(xlims)
# plt.xlabel("Feature")
# plt.ylabel("Coefficient magnitude")

# plt.ylim(-5, 5)
# plt.legend(loc=3)

#----------------------------------------------------------------------------------------------------------------------------
# 決定木の複雑さの制御
cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# # 深さに制約をかけて決定木が複雑にならないように枝刈り
# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)

# with open("tree.dot",'r',encoding='utf-8') as f:
#     dot_graph = f.read()
# print("Feature importances: {}/n".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

# plot_feature_importances_cancer(tree)

# tree = mglearn.plots.plot_tree_not_monotone()

#----------------------------------------------------------------------------------------------------------------------------
# 100個の決定木を使用したランダムフォレスト
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

# print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# 特徴量の重要度
plot_feature_importances_cancer(forest)

#----------------------------------------------------------------------------------------------------------------------------
#　勾配ブースティング回帰木
# デフォルトでは乱数性はないが、強力な事前枝刈りが行われる
# 深さがすごく浅い決定木を用いる
# それぞれの決定木は一部に対してしかいい結果は行えないので、決定木を繰り返し追加して性能を向上させる
# デフォルトでは深さ3の決定木100個作られ、学習率は0.1
# 腫瘍データに勾配ブースティング回帰木を適用させる
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# # 事前枝刈りする場合の精度
# gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
# gbrt.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# # 学習率を下げた場合の精度
# gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
# gbrt.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# # 特徴量の重要度
# gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
# gbrt.fit(X_train, y_train)
# plot_feature_importances_cancer(gbrt)

#----------------------------------------------------------------------------------------------------------------------------
# MLPを適用させてみる
print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

# 訓練セットの特徴量ごとの平均値を算出
mean_on_train = X_train.mean(axis=0)
# 訓練セットの特徴量ごとの標準偏差を算出
std_on_train = X_train.std(axis=0)
# 平均を引き、標準偏差の逆数でスケール変換する
# これでmean=0、std=1になる
X_train_scaled = (X_train - mean_on_train) / std_on_train
# テストセットにも同じ変換をする
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
# 学習繰り返しの回数が最大に達したという警告が出る
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# 学習繰り返しの回数を増やす
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# alphaを1に上げて正規化を強化
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# ニュートラルネットワークの第一層の重みを表したヒートマップ
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

