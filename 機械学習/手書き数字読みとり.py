from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 手書き数字データをダウンロード(28*28の2次元配列)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_labels[0:10])

# 1次元配列に変換
train_images = train_images.reshape((train_images.shape[0], 784))
test_images = test_images.reshape((test_images.shape[0], 784))

# to_categoricalを使ってある要素のみが1で他が0になるようにする(One-Hot表現)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# モデル作成
model = Sequential()
# 256個のユニットを作成
# 活性化関数にシグモイド関数を利用
# input_shape=(784,)は、入ってくるデータの数
model.add(Dense(256, activation='sigmoid', input_shape=(784,))) # 入力層
model.add(Dense(128, activation='sigmoid')) # 隠れ層:複雑な特徴を捉えることが可能に
# rateで無効化率を定める
model.add(Dropout(rate=0.5)) # ドロップアウト:過学習を防ぐために、一部のユニットを無効にする
# 0~9までの多クラス分類になるのでユニット数は10
model.add(Dense(10, activation='softmax')) # 出力層

# モデルのコンパイル
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])# 損失関数,最適化関数,評価指標の設定