from flask import Flask, request, jsonify
import os
import json
import torch
import threading
from firebase_admin import credentials, firestore, initialize_app
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# โหลด Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")

# โหลด Firebase Credentials
cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
firebase_app = initialize_app(cred)
db = firestore.client()

# ✅ โหลด WangchanBERTA Model (ลดขนาดเป็น float16 และใช้ GPU ถ้ามี)
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=5, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32  # ใช้ Half-Precision บน GPU
).to(device)

# ✅ ฟังก์ชันเคลียร์ CUDA Memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 Cleared Unused CUDA Memory")

# สร้าง Flask App
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

# ✅ รันแอป
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
