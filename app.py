import os
import json
import torch
import threading
import requests
import zipfile
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from flask import Flask, request

# ✅ ตั้งค่าตัวแปรสำคัญ
FIREBASE_JSON_URL = "https://firebasestorage.googleapis.com/v0/b/esi-triage-bot-ab4ac.firebasestorage.app/o/esi-triage-bot-ab4ac-firebase-adminsdk-fbsvc-4722ca62ea.json?alt=media"
MODEL_ZIP_URL = "https://firebasestorage.googleapis.com/v0/b/esi-triage-bot-ab4ac.firebasestorage.app/o/esi_model_clean.zip?alt=media"
MODEL_DIR = "./esi_model"
MODEL_PATH = f"{MODEL_DIR}/pytorch_model_fp16.bin"
FIREBASE_CREDENTIALS = "esi-triage-bot-ab4ac-firebase-adminsdk-fbsvc-4722ca62ea.json"

# ✅ ดาวน์โหลด Firebase Credentials
if not os.path.exists(FIREBASE_CREDENTIALS):
    print("⬇️ Downloading Firebase Credentials...")
    r = requests.get(FIREBASE_JSON_URL)
    with open(FIREBASE_CREDENTIALS, "wb") as f:
        f.write(r.content)
    print("✅ Firebase Credentials Downloaded!")

# ✅ ตั้งค่าตัวแปรสำหรับ Firebase
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = FIREBASE_CREDENTIALS

# ✅ โหลด Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

# ✅ ดาวน์โหลดและแตกไฟล์โมเดลจาก Firebase Storage
if not os.path.exists(MODEL_DIR):
    print("⬇️ Downloading model...")
    r = requests.get(MODEL_ZIP_URL)
    with open("model.zip", "wb") as f:
        f.write(r.content)
    print("✅ Model Downloaded! Extracting...")
    with zipfile.ZipFile("model.zip", "r") as zip_ref:
        zip_ref.extractall("./")
    os.remove("model.zip")
    print("✅ Model Extracted!")

# ✅ โหลดโมเดล (ใช้ FP16 ถ้ามี GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=5,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ✅ ฟังก์ชันเคลียร์ CUDA Memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 Cleared Unused CUDA Memory")

# ✅ สร้าง Flask App
app = Flask(__name__)
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ✅ ฟังก์ชันประเมิน ESI
def classify_esi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_esi = torch.argmax(outputs.logits, dim=1).item() + 1
    clear_cuda_memory()  # เคลียร์ CUDA Memory หลังคำนวณเสร็จ
    return predicted_esi

# ✅ Webhook สำหรับรับข้อความจาก LINE
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "No Signature")
    body = request.get_data(as_text=True)
    print(f"📩 Received Webhook: {body}")
    if not signature:
        return "Missing Signature", 400
    threading.Thread(target=handler.handle, args=(body, signature)).start()
    return "OK", 200  # ตอบกลับทันที

# ✅ ฟังก์ชันตอบกลับข้อความจาก LINE
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
