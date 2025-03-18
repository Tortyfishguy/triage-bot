from transformers import pipeline

# โหลดโมเดล MedGPT จาก Hugging Face
MODEL_NAME = "ชื่อโมเดลของ MedGPT"  # แทนที่ด้วยชื่อโมเดลที่ถูกต้อง
esi_pipeline = pipeline("text-generation", model=MODEL_NAME)

def classify_esi(text):
    """
    ฟังก์ชันวิเคราะห์อาการที่พิมพ์มา และจัดระดับ ESI (Emergency Severity Index)
    """
    prompt = f"Patient symptoms: {text}\nWhat is the ESI level (1-5)?"
    response = esi_pipeline(prompt, max_length=200)
    
    # แปลงผลลัพธ์ให้เป็นระดับ ESI
    esi_level = extract_esi_level(response[0]['generated_text'])
    return esi_level

def extract_esi_level(text):
    """
    ดึงค่า ESI level ออกจากข้อความที่โมเดลให้มา
    """
    for level in range(1, 6):
        if f"ESI {level}" in text or f"ESI-{level}" in text:
            return level
    return 5  # ถ้าไม่พบให้ถือว่าเป็น ESI 5 (ไม่ร้ายแรง)

