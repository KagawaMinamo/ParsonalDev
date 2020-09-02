import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer

# データセットの生成
# forgeデータセットは二つの特徴量を持つ
X, y = mglearn.datasets.make_forge()
# データセットのプロット
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 凡例の表示
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature") # 第一特徴量
plt.ylabel("Second feature") # 第二特徴量
print("X.shape: {}".format(X.shape))

# waveで回帰アルゴリズム
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')# 第3引数はオプションを表しており、今回はmarkerを設定している
plt.ylim(-3, 3)
plt.xlabel("Feature")# 特徴量
plt.ylabel("Target")# 回帰ターゲット