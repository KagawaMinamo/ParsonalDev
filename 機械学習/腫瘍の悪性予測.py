import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

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
# k-最近傍法によるクラス分類
# 星印で示される3つの新しいデータポイント
# 1-最近傍法での予測では近傍点のラベルが予測されたラベルになる
mglearn.plots.plot_knn_classification(n_neighbors=1)
# 個々のテストする点に対して、近傍点のうちいくつがクラス0に属し、いくつがクラス1に属すか計算し、多いほうのクラスをその点に与える
mglearn.plots.plot_knn_classification(n_neighbors=3)

