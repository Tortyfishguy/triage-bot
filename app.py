import os
import threading
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from model import classify_esi

# ✅ โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# ✅ สร้าง Flask App
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ✅ Webhook รับข้อความจาก LINE
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "No Signature")
    body = request.get_data(as_text=True)
    threading.Thread(target=handler.handle, args=(body, signature)).start()
    return "OK", 200

# ✅ ฟังก์ชันตอบกลับ LINE
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    esi_level, response_text = classify_esi(user_text)

    # ✅ กำหนดคำตอบตามระดับ ESI
    if esi_level in [1, 2]:
        reply_text = f"🚨 {response_text} (ESI {esi_level})"
    elif esi_level == 3:
        reply_text = f"🩺 {response_text} (ESI {esi_level})"
    else:
        reply_text = f"💊 {response_text} (ESI {esi_level})"

    threading.Thread(target=lambda: line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))).start()

# ✅ รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
