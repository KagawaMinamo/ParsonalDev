from PIL import Image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sys

file_path = ''
file_save_path = ''

# ファイル選択する処理
def file_select(event):
  # 初期化
  txt_file_path.delete(0, tk.END)
  # 拡張子
  fTyp = [('JPGファイル', '*.jpg'),('PNGファイル','*.png'),('GIFファイル','*.gif')]
  # 最初の表示するディレクトリ
  idir = 'C:\\Users'
  file_path = tk.filedialog.askopenfilename(filetypes = fTyp, initialdir = idir)

  # ファイルパスが格納されていたらテキストボックスに表示
  if file_path:
     txt_file_path.insert(tk.END, file_path)
  else:
     return

def file_save(event):
  txt_file_save_path.delete(0, tk.END)
  # 拡張子
  fTyp = [("すべてのファイル", "*")]
  # 最初の表示するディレクトリ
  idir = 'C:\\Users'
  file_save_path = tk.filedialog.asksaveasfilename(defaultextension = 'jpg', filetypes = fTyp, initialdir = idir)
  # ファイルパスが格納されていたらテキストボックスに表示
  if file_save_path:
     txt_file_save_path.insert(tk.END, file_save_path )
  else:
     return


  # 画像をぼかす処理
def Img_Blur(event):
  # ファイルを選んでなかったらメッセージ出して処理終了
  if txt_file_path.get() is '':
    messagebox.showerror('エラー','画像ファイルを選択してください')
    return

  if txt_file_save_path.get() is '':
    messagebox.showerror('エラー','保存先を選択してください')
    return

  #例外処理
  try:
    # 画像の読み込み(自分で撮影したやつ)
    img = Image.open(txt_file_path.get())

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
  except Exception as ex:
    messagebox.showerror('エラー', ex)
    return
  
  # 保存
  clone_img.save(txt_file_save_path.get())
  messagebox.showinfo('情報', 'ぼかし処理が完了しました')


# モノクロ処理  
def Img_Monochrome(event):
  # ファイルを選んでなかったらメッセージ出して処理終了
  if txt_file_path.get() is '':
    messagebox.showerror('エラー','画像ファイルを選択してください')
    return

  if txt_file_save_path.get() is '':
    messagebox.showerror('エラー','保存先を選択してください')
    return

  img = Image.open(txt_file_path.get())
  # グレースケールに変換
  gray_img = img.convert('L')
  # 保存
  gray_img.save(txt_file_save_path.get())
  messagebox.showinfo('情報', 'モノクロ処理が完了しました')


# 色反転処理
def Img_RGB_Flip(event):
  if txt_file_path.get() is '':
    messagebox.showerror('エラー','画像ファイルを選択してください')
    return

  if txt_file_save_path.get() is '':
    messagebox.showerror('エラー','保存先を選択してください')
    return

  img = Image.open(txt_file_path.get())
  width, height = img.size
  flip_img = Image.new('RGB', (width, height))
  img_pixels = np.array([[img.getpixel((x,y)) for x in range(width)] for y in range(height)])

  # 色を反転する
  reverse_color_pixels = 255 - img_pixels
  for y in range(height):
    for x in range(width):
      # 反転した色の画像を作成する
      r,g,b = reverse_color_pixels[y][x]
      flip_img.putpixel((x,y), (r,g,b))

  flip_img.save(txt_file_save_path.get())
  messagebox.showinfo('情報', '色反転処理が完了しました')


def Close(event):
    sys.exit(0)

#-------------------------------------------------------------------------------------------------

root = tk.Tk()
root.title(u"画像処理アプリ")
# ウィンドウのwidthとheight
root.geometry("480x270")

# ファイル選択ラベル
lbl_file_path = tk.Label(text=u'ファイル名')
lbl_file_path.place(x=10, y=10)

# 処理対象ファイルパスを表示するテキストボックス
txt_file_path = tk.Entry(width = 55)
txt_file_path.place(x = 10, y = 40)

# ファイル選択ボタン
btn_file_select = tk.Button(text=u'ファイル選択')
btn_file_select.bind("<Button-1>", file_select) # クリックイベントにメソッドをバインド
btn_file_select.place(x=360, y=35)

# 保存場所ラベル
lbl_file_path = tk.Label(text=u'保存場所')
lbl_file_path.place(x=10, y=65)

# 処理対象ファイルパスを表示するテキストボックス
txt_file_save_path = tk.Entry(width = 55)
txt_file_save_path.place(x = 10, y = 90)

# 保存場所選択ボタン
btn_file_save_select = tk.Button(text=u'保存場所選択')
btn_file_save_select.bind("<Button-1>", file_save) # クリックイベントにメソッドをバインド
btn_file_save_select.place(x=360, y=85)

# ぼかし処理ボタン
btn_blur = tk.Button(text=u'ぼかす')
btn_blur.bind("<Button-1>", Img_Blur) 
btn_blur.place(x=100, y=180)

# モノクロ処理ボタン
btn_mosaic = tk.Button(text=u'モノクロ')
btn_mosaic.bind("<Button-1>", Img_Monochrome) 
btn_mosaic.place(x=200, y=180)

# 処理ボタン
btn_blur = tk.Button(text=u'色反転')
btn_blur.bind("<Button-1>", Img_RGB_Flip) 
btn_blur.place(x=300, y=180)

# 注意文言ラベル
lbl_caution = tk.Label(text=u'●保存場所を選択するときに拡張子を入力しない！')
lbl_caution.place(x=10, y=130)

lbl_caution = tk.Label(text=u'●保存される画像はJPEG形式！')
lbl_caution.place(x=10, y=150)

# 閉じるボタン
btn_close = tk.Button(text=u'閉じる')
btn_close.bind("<Button-1>", Close) 
btn_close.place(x=380, y=230)

root.mainloop()
