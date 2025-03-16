import os
import json
import torch
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔹 โหลดตัวแปรจาก Environment Variables (เก็บ API Key ให้ปลอดภัย)
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")
print("FIREBASE_CREDENTIALS_JSON:", FIREBASE_CREDENTIALS_JSON[:100]) # Debug

# 🔹 โหลด Firebase Credentials จาก Environment Variables
cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
firebase_app = initialize_app(cred)
db = firestore.client()

# 🔹 โหลดโมเดล WangchanBERTa
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

# 🔹 สร้าง Flask App
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 🔹 ฟังก์ชันวิเคราะห์ ESI
def classify_esi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_esi = torch.argmax(outputs.logits, dim=1).item() + 1
    return predicted_esi

# 🔹 Route สำหรับรับ Webhook จาก LINE
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except Exception as e:
        return str(e), 400
    
    return "OK"

# 🔹 จัดการข้อความจาก LINE
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text
    esi_level = classify_esi(text)
    
    response_text = f"🔹 ระดับ ESI จากอาการ: {esi_level}"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))

# 🔹 รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
