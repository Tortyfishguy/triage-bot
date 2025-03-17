from flask import Flask, request
import os
import json
import torch
import threading
import requests
import zipfile

from firebase_admin import credentials, firestore, initialize_app, storage
from google.cloud import storage as gcs_storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ✅ ตั้งค่า Firebase Storage URL (แก้ไขให้เป็นของโปรเจคคุณ)
FIREBASE_JSON_URL = "https://firebasestorage.googleapis.com/v0/b/esi-triage-bot-ab4ac.firebasestorage.app/o/esi-triage-bot-ab4ac-firebase-adminsdk-fbsvc-4722ca62ea.json?alt=media"
MODEL_ZIP_URL = "https://firebasestorage.googleapis.com/v0/b/esi-triage-bot-ab4ac.firebasestorage.app/o/esi_model_clean.zip?alt=media"

# ✅ ตั้งค่าที่เก็บไฟล์บนเซิร์ฟเวอร์
LOCAL_CREDENTIALS_PATH = "/tmp/firebase-adminsdk.json"
LOCAL_MODEL_PATH = "/tmp/esi_model"

# ✅ ดึงไฟล์ Firebase Credentials JSON จาก Firebase Storage
def download_firebase_credentials():
    if not os.path.exists(LOCAL_CREDENTIALS_PATH):
        print("🔽 Downloading Firebase Credentials JSON...")
        response = requests.get(FIREBASE_JSON_URL)
        with open(LOCAL_CREDENTIALS_PATH, "wb") as file:
            file.write(response.content)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = LOCAL_CREDENTIALS_PATH
        print("✅ Firebase Credentials Downloaded")

# ✅ ดึงไฟล์โมเดลจาก Firebase Storage
def download_and_extract_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("🔽 Downloading Model ZIP...")
        response = requests.get(MODEL_ZIP_URL)
        zip_path = "/tmp/esi_model.zip"
        with open(zip_path, "wb") as file:
            file.write(response.content)
        
        print("📦 Extracting Model...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("/tmp/")
        print("✅ Model Extracted")

# ✅ โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# ✅ โหลด Firebase Credentials
download_firebase_credentials()
cred = credentials.Certificate(LOCAL_CREDENTIALS_PATH)
firebase_app = initialize_app(cred)
db = firestore.client()

# ✅ โหลดโมเดล (ลดขนาดเป็น FP16 และใช้ GPU ถ้ามี)
download_and_extract_model()
MODEL_PATH = "/tmp/esi_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=5,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32  # ใช้ Half-Precision บน GPU
).to(device)

# ✅ ฟังก์ชันเคลียร์ CUDA Memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 Cleared Unused CUDA Memory")

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

    # เคลียร์ CUDA Memory หลังคำนวณเสร็จ
    clear_cuda_memory()

    return predicted_esi

# ✅ Webhook ตอบกลับทันที + ใช้ Threading
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "No Signature")
    body = request.get_data(as_text=True)

    print(f"📩 Received Webhook: {body}")
    print(f"🔐 Signature: {signature}")

    if not signature:
        print("❌ Missing X-Line-Signature")
        return "Missing Signature", 400

    # ใช้ Thread เพื่อให้ Webhook ตอบกลับทันที
    def handle_message_async():
        try:
            handler.handle(body, signature)
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")

    threading.Thread(target=handle_message_async).start()
    
    return "OK", 200  # ตอบกลับทันทีเพื่อป้องกัน timeout

# ✅ ตอบกลับจาก LINE
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text
    esi_level = classify_esi(text)  # ใช้ฟังก์ชันประเมินระดับ ESI

    # แปลงระดับ ESI เป็นข้อความที่เข้าใจง่าย
    if esi_level in [1, 2]:
        response_text = f"🚨 อาการของคุณจำเป็นต้องเข้ารับการรักษาที่ห้องฉุกเฉินทันที! (ESI {esi_level})"
    elif esi_level == 3:
        response_text = f"🩺 ควรได้รับการประเมินโดยแพทย์ (ESI {esi_level})"
    else:
        response_text = f"💊 แนะนำให้เข้ารับการตรวจที่โรงพยาบาลในวันถัดไป (ESI {esi_level})"

    # ✅ ใช้ Threading เพื่อให้ LINE ตอบกลับเร็วขึ้น
    def reply():
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))

    thread = threading.Thread(target=reply)
    thread.start()

# ✅ รันแอป (ใช้ Uvicorn Worker เพื่อลด RAM)
if __name__ == "__main__":
    from uvicorn import run
    run(app, host="0.0.0.0", port=10000, workers=1)
