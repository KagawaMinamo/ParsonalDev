import numpy as np
import matplotlib.pyplot
import pandas as pd
from IPython.display import display
from sklearn.datasets import load_iris

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
