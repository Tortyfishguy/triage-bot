import os
from transformers import pipeline

# โหลด Hugging Face API Token จาก Environment Variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ตรวจสอบว่าได้ Token มาหรือไม่
if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ Missing HUGGINGFACE_TOKEN. Please set it in the environment variables.")

# ชื่อโมเดลของ Perceptor AI ที่ต้องการใช้
MODEL_NAME = "Perceptor-AI/perceptor-medical-qa"

# โหลดโมเดลผ่าน pipeline โดยเพิ่ม Token สำหรับ Authentication
qa_pipeline = pipeline(
    "text-generation",
    model=MODEL_NAME,
    token=HUGGINGFACE_TOKEN
)

def classify_esi(text):
    """
    ใช้โมเดล Perceptor AI วิเคราะห์ข้อความสุขภาพ และแปลงผลลัพธ์เป็นระดับ ESI
    """
    response = qa_pipeline(text, max_length=100, truncation=True)
    generated_text = response[0]["generated_text"]

    # แปลงผลลัพธ์เป็นระดับ ESI (กำหนดตาม heuristic)
    if "emergency" in generated_text.lower():
        return 1  # ESI 1 (วิกฤติ)
    elif "urgent" in generated_text.lower():
        return 2  # ESI 2 (เร่งด่วน)
    elif "moderate" in generated_text.lower():
        return 3  # ESI 3 (ทั่วไป)
    else:
        return 4  # ESI 4-5 (ไม่ฉุกเฉิน)
