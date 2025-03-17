from flask import Flask, request, jsonify
import os
import json
import torch
import threading
import zipfile
from firebase_admin import credentials, firestore, initialize_app
from google.cloud import storage  # ✅ Firebase Storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ✅ โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")
FIREBASE_BUCKET_NAME = os.getenv("FIREBASE_BUCKET_NAME")

# ✅ โหลด Firebase Credentials
cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
firebase_app = initialize_app(cred)
db = firestore.client()

# ✅ ตั้งค่า Firebase Storage
storage_client = storage.Client()
bucket = storage_client.bucket(FIREBASE_BUCKET_NAME)

# ✅ ฟังก์ชันโหลดโมเดลจาก Firebase Storage
def download_and_extract_model():
    model_zip_path = "esi_model.zip"
    extract_to = "./esi_model"

    # ✅ ดาวน์โหลดไฟล์จาก Firebase Storage
    blob = bucket.blob(model_zip_path)
    blob.download_to_filename(model_zip_path)
    print(f"📥 ดาวน์โหลด {model_zip_path} สำเร็จ")

    # ✅ แตกไฟล์ ZIP
    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"📂 แตกไฟล์ {model_zip_path} ไปที่ {extract_to} สำเร็จ")

    # ✅ ลบไฟล์ ZIP หลังแตกไฟล์เสร็จ
    os.remove(model_zip_path)
    print("🗑 ลบไฟล์ ZIP สำเร็จ")

# ✅ ดาวน์โหลดและแตกไฟล์โมเดลก่อนเริ่มแอป
download_and_extract_model()

# ✅ ตั้งค่าการใช้ GPU หรือ CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ โหลดโมเดลจากโฟลเดอร์ที่ดาวน์โหลดมา
MODEL_PATH = "./esi_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=5,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # ใช้ Half-Precision บน GPU
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

    # ✅ เคลียร์ CUDA Memory หลังคำนวณเสร็จ
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

    # ✅ ใช้ Thread เพื่อให้ Webhook ตอบกลับทันที
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

    # ✅ แปลงระดับ ESI เป็นข้อความที่เข้าใจง่าย
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

# ✅ รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
