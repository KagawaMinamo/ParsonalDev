import urllib
from bs4 import BeautifulSoup
import requests
from datetime import datetime

#天気を取得するRSS(大阪府)
weather_rss = 'https://rss-weather.yahoo.co.jp/rss/days/27.xml'
#警報・注意報を取得するRSS(大阪府全体)
warn_rss = 'https://rss-weather.yahoo.co.jp/rss/warn/27.xml'

line_notify_token = 'gaZkcCgMXR0Ro01QQm5DqT5kSm646HlMSaf6jJnZhZ4'   #アクセストークン
line_notify_api = 'https://notify-api.line.me/api/notify'

tenki = []
detail = []

#WebページのHTMLタグから情報をスクレイピングする
def WeatherParser(rss):
   with urllib.request.urlopen(rss) as res:
      xml = res.read()
      soup = BeautifulSoup(xml, "html.parser")
      for item in soup.find_all("item"):
         title = item.find("title").string                  #タイトル
         description = item.find("description").string      #説明
         pubdate = item.find("pubdate").string              #公開日

         #スクレイピングしたデータを格納
         if title.find("[ PR ]") == -1:
            tenki.append(title)
            detail.append(description)

#天気予報サイトのHTMLタグから天気情報を抽出
WeatherParser(weather_rss) 

n = len(tenki)

#情報をLINEに出力
def Scraping():
    for i in range(0,n):
        message = tenki[i]
        payload = {'message':"\n" + message}
        headers = {'Authorization': 'Bearer ' + line_notify_token}
    
        #情報を出力
        line_notify = requests.post(line_notify_api, data = payload, headers = headers)

Scraping()

#リスト初期化
tenki = []
detail = []

# 天気予報サイトのHTMLタグから警報・注意報を抽出
WeatherParser(warn_rss) 
n = len(tenki)
Scraping()

#情報をLINEに出力
url = 'https://weather.yahoo.co.jp/weather/jp/27/6200.html'     #天気Webページ(大阪府)
payload = {'message': url}
headers = {'Authorization': 'Bearer ' + line_notify_token}      # NotifyのURL
line_notify = requests.post(line_notify_api, data=payload, headers=headers) 