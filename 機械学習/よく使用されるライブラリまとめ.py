import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

# Numpyは多次元配列や線形代数、フーリエ変換、疑似乱数生成などの数学関数が用意されている
arr = np.array([[1, 2, 3],[4, 5, 6]])
print("arr:\n{}".format(arr))

# SciPyは高度な線形代数ルーチンや数学関数の最適化、信号処理、特殊な数学関数、統計分布などの機能を持つ
# 疎行列の表現
eye = np.eye(4)
sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n{}".format(sparse_matrix))

# matplotlibは折れ線グラフ、ヒストグラム、散布図など科学技術計算向けのグラフ描画ライブラリ
# -10から10までを100ステップに区切った列を配列として生成
x = np.linspace(-10, 10, 100)
# サイン関数を用いて二つ目の配列を生成
y = np.sin(x)
# 一方の配列に対して他方の配列をプロット
plt.plot(x, y, marker="x")

# pandasはデータを変換したり解析したりするためのライブラリで、作成したテーブルを変更する関数や操作する様々な手法を提供している
# 簡単なデータセット作成
data = {'name':["John", "Anna", "Peter", "Linda"],
        'Location':["New York", "Paris", "Berlin", "London"],
        'Age':[24, 13, 53, 33]
        }
data_pandas = pd.DataFrame(data)
# データセット表示
print(data_pandas)