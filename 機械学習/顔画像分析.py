import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from IPython.display import display
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler # スケール変換器
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier

# 固有顔による特徴量抽出
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

# 入っている顔データ
# 枚数、ピクセル
print("people.images.shape: {}".format(people.images.shape))
# クラスの数
print("Number of classes: {}".format(len(people.target_names)))

# 各ターゲットの出現回数をカウント
counts = np.bincount(people.target)
# ターゲット名と出現回数を並べて表示
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()
    
# 偏りを減らすために、各人の画像を50に制限
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]
# 0から255で表現されている、グレースケールの値0と1の間に変換
# 数値的に安定する
X_people = X_people / 255.

# 訓練セットとテストセットにデータを分割
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# KNeighborsClassifierを1-最近傍で構築
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))