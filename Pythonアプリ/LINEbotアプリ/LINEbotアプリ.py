from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
import os

app = Flask(__name__)

#環境変数取得
#アクセストークン
YOUR_CHANNEL_ACCESS_TOKEN = os.environ['tYJZFUtqrCVgvsrZVEUVjbGMe1m9OFeQ8U4jKliH7bllRHdv524AImtJ1gmRfFfLJ4UIKQbV9qpEdvY8DEjSsHA6DYF/7lp7C7MJRPe4GTHUR3XFB7ujkz7FXRPbuRMH8iZmhlnBJ7tpIhWCgVCGgQdB04t89/1O/w1cDnyilFU=']
#ChannelSecret
YOUR_CHANNEL_SECRET = os.environ['d9f8bcc8a388525d9c24a8f3f15fe9c7']

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    #例外処理
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))


if __name__ == "__main__":
#    app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
