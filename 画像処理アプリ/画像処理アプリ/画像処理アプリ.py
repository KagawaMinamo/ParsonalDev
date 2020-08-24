from PIL import Image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

file_path = ''
file_save_path = ''

# ファイル選択する処理
def file_select(event):
  # 拡張子
  fTyp = [("", "*")]
  # 最初の表示するディレクトリ
  idir = 'C:\\Users'
  file_path = tk.filedialog.askopenfilename(filetypes = fTyp, initialdir = idir)

  # ファイルパスが格納されていたらテキストボックスに表示
  if file_path:
     txt_file_path.insert(tk.END, file_path)
  else:
     return

  # ファイルを保存する処理
def file_save(event):
  # 拡張子
  fTyp = [("画像", ".jpg")]
  # 最初の表示するディレクトリ
  idir = 'C:\\Users'
  file_save_path = tk.filedialog.asksaveasfilename(filetypes = fTyp, title = "保存場所を選択", initialdir = idir)

  # ファイルパスが格納されていたらぼかし処理実行
  if file_save_path:
     Img_Blur
  else:
     return
 

  # 画像をぼかす処理
def Img_Blur():
  #実行中のディレクトリ名の取得
  img_directory_path = os.path.dirname(__file__)

  # 画像の読み込み(自分で撮影したやつ)
  #img = Image.open(img_directory_path + '\画像\伏見稲荷大社.jpg')
  img = Image.open(file_path)

  # 画像の高さと幅を取得
  width, height = img.size
  filter_size = 30
  # Imageオブジェクトを作成する
  clone_img = Image.new('RGB', (width - filter_size, height - filter_size))
  #ピクセルの色を取得し配列に格納
  img_pixels = np.array([[img.getpixel((x,y)) for x in range(width)] for y in range(height)])

  for y in range(height - filter_size):
    for x in range(width - filter_size):
      # 位置(x,y)を起点に縦横フィルターサイズの小さい画像をオリジナル画像から切り取る
      partial_img = img_pixels[y:y + filter_size, x:x + filter_size]
      # 小さい画像の各ピクセルの値を一列に並べる
      color_array = partial_img.reshape(filter_size ** 2, 3)
      # 各R,G,Bそれぞれの平均を求めて加工後画像の位置(x,y)のピクセルの値にセットする
      mean_r, mean_g, mean_b = color_array.mean(axis = 0)
      clone_img.putpixel((x,y), (int(mean_r), int(mean_g), int(mean_b)))

  #画像の保存
  #clone_img.show()
  #clone_img.save(img_directory_path + '\画像\伏見稲荷大社(ぼかし).jpg')
  clone_img.save(file_save_path)


root = tk.Tk()
root.title(u"画像処理アプリ")
root.geometry("400x300")

# テキストボックス
txt_file_path = tk.Entry()
txt_file_path.pack()

# ファイル選択ボタン
btn_file_select = tk.Button(text=u'ファイル選択')
btn_file_select.bind("<Button-1>", file_select) # クリックイベントにメソッドをバインド
btn_file_select.pack()

# ぼかし処理ボタン
btn_file_select = tk.Button(text=u'ぼかす')
btn_file_select.bind("<Button-1>", Img_Blur) 
btn_file_select.pack()

root.mainloop()
