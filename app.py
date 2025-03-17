import os
import json
import torch
import threading
import requests
import zipfile

from flask import Flask, request, jsonify
from firebase_admin import credentials, initialize_app, storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ✅ โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL")  # ลิงก์ไฟล์ .zip ของโมเดลจาก Firebase Storage

# ✅ ตั้งค่าการใช้ Firebase
cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
firebase_app = initialize_app(cred, {"storageBucket": "esi-triage-bot-ab4ac.appspot.com"})

# ✅ ตั้งค่าการใช้ CPU เท่านั้น
device = "cpu"

# ✅ ฟังก์ชันโหลดและแตกไฟล์โมเดลจาก Firebase Storage
MODEL_DIR = "./esi_model"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_zip_path = "esi_model.zip"

    print("📥 Downloading model from Firebase Storage...")
    response = requests.get(MODEL_ZIP_URL)
    with open(model_zip_path, "wb") as f:
        f.write(response.content)
    
    print("📂 Extracting model...")
    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    
    os.remove(model_zip_path)
    print("✅ Model is ready!")

# ✅ โหลดโมเดล
download_and_extract_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, 
    num_labels=5,
    torch_dtype=torch.float32
).to(device)

# ✅ ฟังก์ชันเคลียร์หน่วยความจำที่ไม่ใช้
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ✅ สร้าง Flask App
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ✅ ฟังก์ชันประเมิน ESI
def classify_esi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_esi = torch.argmax(outputs.logits, dim=1).item() + 1
    clear_memory()
    return predicted_esi

# ✅ Webhook รับข้อความจาก LINE
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "No Signature")
    body = request.get_data(as_text=True)

    print(f"📩 Received Webhook: {body}")
    print(f"🔐 Signature: {signature}")

    if not signature:
        print("❌ Missing X-Line-Signature")
        return "Missing Signature", 400

    threading.Thread(target=handler.handle, args=(body, signature)).start()
    return "OK", 200  # ตอบกลับทันที

# ✅ ฟังก์ชันตอบกลับ LINE
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text
    esi_level = classify_esi(text)

    if esi_level in [1, 2]:
        response_text = f"🚨 อาการของคุณจำเป็นต้องเข้ารับการรักษาที่ห้องฉุกเฉินทันที! (ESI {esi_level})"
    elif esi_level == 3:
        response_text = f"🩺 ควรได้รับการประเมินโดยแพทย์ (ESI {esi_level})"
    else:
        response_text = f"💊 แนะนำให้เข้ารับการตรวจที่โรงพยาบาลในวันถัดไป (ESI {esi_level})"

    threading.Thread(target=lambda: line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))).start()

# ✅ รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
