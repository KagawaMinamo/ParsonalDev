from PIL import Image
import numpy as np
import os

#実行中のディレクトリ名の取得
img_directory_path = os.path.dirname(__file__)

print(img_directory_path)

# 画像の読み込み(自分で撮影したやつ)
img = Image.open(img_directory_path + '\画像\伏見稲荷大社.jpg')
#オリジナル画像の幅と高さを取得
width, height = img.size

filter_size = 20

# Imageオブジェクトを作成する
clone_img = Image.new('RGB', (width - filter_size, height - filter_size))

img_pixels = []
#縦
for y in range(height):
  #横
  for x in range(width):
    # ピクセルの色を取得し、img_pixelsに追加する
    img_pixels.append(img.getpixel((x,y)))
# あとで計算しやすいようにnumpyのarrayに変換しておく
img_pixels = np.array(img_pixels)


for y in range(height - filter_size):
  for x in range(width - filter_size):
    # 位置(x,y)を起点に縦横フィルターサイズの小さい画像をオリジナル画像から切り取る            
    partial_img = img_pixels[y:y + filter_size, x:x + filter_size]

    # 小さい画像の各ピクセルの値を一列に並べる
    color_array = partial_img.reshape(filter_size ** 2, 3)

    # 各R,G,Bそれぞれの平均を求めて加工後画像の位置(x,y)のピクセルの値にセットする
    mean_r, mean_g, mean_b = color_array.mean(axis = 0)
    clone_img.putpixel((x,y), (int(mean_r), int(mean_g), int(mean_b)))

# 画像の保存
clone_img.show()
clone_img.save(img_directory_path + '\画像\伏見稲荷大社(ぼかし).jpg')

print(img_directory_path + '\画像\伏見稲荷大社(ぼかし).jpg')
