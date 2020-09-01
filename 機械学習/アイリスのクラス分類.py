import numpy as np
#import matplotlib.pyplot
import pandas as pd
#from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#import mglearn
from sklearn.neighbors import KNeighborsClassifier

# どんなデータか見る
# データの読み込み
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# データセットの簡単な説明(一部のみ表示)
print(iris_dataset['DESCR'][:193]+"\n...")
# 予測しようとしている花の種類
print("Target names: {}".format(iris_dataset['target_names']))
# それぞれの特徴量の説明が格納されている
print("Feature names: \n{}".format(iris_dataset['feature_names']))
# ガクの長さ、ガクの幅、花弁の長さ、花弁の幅がNumPy配列として格納
print("Type of data: {}".format(type(iris_dataset['data'])))
# 個々の花に対応し、列は個々の花に対して行われた4つの測定に対応する
print("Shape of data: {}".format(iris_dataset['data'].shape))
# 最初の5つのサンプルを見る
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# 測定された個々の花の種類がNumPy配列として格納
print("Type of target: {}".format(type(iris_dataset['target'])))
# targetは1次元配列で、個々の花に1つのエントリが対応する
print("Shape of target: {}".format(iris_dataset['target'].shape))
# 種類は0から2までの整数としてエンコードされる
# 0はsetosa、1はversicolor、2はvirginica
print("Target:\n{}".format(iris_dataset['target']))

#------------------------------------------------------------------------------
# 成功度合いの測定
# Xはデータ、yはラベル(既知の品種名)
# trainは訓練データ(学習モデルの構築に使う)、testはテストデータ(モデルがどの程度機能するかを評価する)
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# データの検査
# iris_dataset.feature_namesを使ってカラムに名前を付ける
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# データフレームからscatter matrixを作成し、y_trainに従って色を付ける
grr = pd.plotting.scatter_matrix(iris_dataframe, c = y_train ,figsize = (15, 15), marker = 'o', hist_kwds = {'bins':20}, s = 60, alpha = .8)

# モデル(k-最近傍法)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

# 品種を調べる
# データをNumPy配列に格納し、その形を計算
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

# 予測を行うにはknnオブジェクトのpredictメソッドを呼ぶ
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:", iris_dataset['target_names'][prediction])

# モデルの評価
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# scoreでもテストセットに対する精度を測定できる
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# まとめの部分
# 訓練と評価を行う必要な最小の手順
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))