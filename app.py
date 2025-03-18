import os
import requests
import zipfile
import torch
import threading
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ✅ โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FIREBASE_JSON_URL = os.getenv("FIREBASE_JSON_URL")
MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL")

# ✅ ดาวน์โหลด service-account.json ถ้ายังไม่มี
JSON_PATH = "service-account.json"
if not os.path.exists(JSON_PATH):
    print("📥 Downloading service-account.json from Firebase Storage...")
    response = requests.get(FIREBASE_JSON_URL)
    with open(JSON_PATH, "wb") as f:
        f.write(response.content)
    print("✅ service-account.json is ready!")

# ✅ ตั้งค่า Environment Variable ให้ Firebase ใช้ไฟล์ที่ดาวน์โหลด
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = JSON_PATH

# ✅ ฟังก์ชันโหลดและแตกไฟล์โมเดลจาก Firebase Storage (ถ้ายังไม่มี)
MODEL_DIR = "./esi_model"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_zip_path = "esi_model.zip"

    print("📥 Downloading model from Firebase Storage...")
    response = requests.get(MODEL_ZIP_URL)
    if response.status_code != 200:
        print(f"❌ Failed to download model. Status code: {response.status_code}")
        return False

    with open(model_zip_path, "wb") as f:
        f.write(response.content)

    print("📂 Extracting model...")
    try:
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(model_zip_path)  # ลบไฟล์ zip หลังจากแตกไฟล์เสร็จ
        print("✅ Model is ready!")
        return True
    except zipfile.BadZipFile:
        print("❌ Error: The downloaded file is not a valid zip file!")
        return False

# ✅ โหลดโมเดล
if download_and_extract_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, 
        num_labels=5,
        torch_dtype=torch.float32
    )

# ✅ ฟังก์ชันโหลดโมเดลเฉพาะตอนใช้งาน
def classify_esi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_esi = torch.argmax(outputs.logits, dim=1).item() + 1
    return predicted_esi

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

# ✅ ฟังก์ชันตอบกลับ LINE (คงรูปแบบเดิม)
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
