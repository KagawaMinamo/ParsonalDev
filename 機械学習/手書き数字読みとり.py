from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 手書き数字データをダウンロード(28*28の2次元配列)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データが入っているか確認
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

# 学習
# batch_sizeで訓練データをいくつの単位で訓練に利用する設定、数値が大きいほど学習速度は上がるがメモリを消費する
# epochsで訓練するエポック数
# validation_splitで訓練データと検証データを分ける割合
# lossは訓練データの誤差,0に近いほど良い結果
# accは訓練データの正解率,1に近いほど良い結果
# val_lossは検証データの誤差,0に近いほど良い結果
# val_accは検証データの正解率,1に近いほど良い結果
history = model.fit(train_images, train_labels, batch_size=500, epochs=5, validation_split=0.2)

# 学習結果グラフの作成
# history.historyで学習結果
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')# y軸のタイトル
plt.xlabel('epoch')# x軸のタイトル
plt.legend(loc='best')
plt.show()

# 学習結果の評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc ))

# 推論
# 推論する画像の表示
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(test_images[i].reshape((28, 28)), 'gray')
plt.show()
# 画像をmodel.predictに入れる
test_predictions = model.predict(test_images[0:10])
test_predictions = np.argmax(test_predictions, axis=1)
print(test_predictions)#画像を読みとった結果