import os
import threading
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from model import classify_esi  # นำเข้าโมเดลจาก model.py

# โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# ตั้งค่า Flask App
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/")
def home():
    return "DeepSeek AI ESI Classifier is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "No Signature")
    body = request.get_data(as_text=True)

    if not signature:
        return "Missing Signature", 400

    threading.Thread(target=handler.handle, args=(body, signature)).start()
    return "OK", 200

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    esi_level = classify_esi(user_message)

    if esi_level in [1, 2]:
        response_text = f"🚨 อาการของคุณรุนแรง ควรรีบไปห้องฉุกเฉินทันที! (ESI {esi_level})"
    elif esi_level == 3:
        response_text = f"🩺 คุณควรพบแพทย์เร็วที่สุดเพื่อตรวจสอบเพิ่มเติม (ESI {esi_level})"
    elif esi_level in [4, 5]:
        response_text = f"💊 อาการของคุณสามารถรอพบแพทย์ที่ OPD ได้ (ESI {esi_level})"
    else:
        response_text = "❌ ไม่สามารถประเมินได้ กรุณาลองอธิบายอาการให้ละเอียดขึ้น"

    threading.Thread(target=lambda: line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))).start()

# รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

