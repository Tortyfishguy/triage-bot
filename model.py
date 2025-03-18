import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ✅ โหลด Environment Variables
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-chat-hf")  # ใช้ Llama 2 (7B)
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", 1024))

# ✅ โหลดโมเดลและ Tokenizer
print("📥 Loading Llama 2 model...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL)

# ✅ ใช้ Pipeline ของ Hugging Face สำหรับ Text Generation
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ ฟังก์ชันวิเคราะห์อาการและจัดอันดับ ESI
def classify_esi(patient_symptoms):
    prompt = f"""
    คุณเป็น AI ผู้เชี่ยวชาญด้านการแพทย์
    ผู้ป่วยบอกว่าพวกเขามีอาการดังต่อไปนี้: "{patient_symptoms}"

    กรุณาประเมินระดับความเร่งด่วนตาม **ESI Level** และบอกว่าผู้ป่วยควรไปพบแพทย์ด่วนหรือไม่:
    - **ESI 1-2:** อาการรุนแรงมาก ควรพบแพทย์ทันที
    - **ESI 3:** ควรพบแพทย์เร็วที่สุด
    - **ESI 4-5:** สามารถรอพบแพทย์ในวันถัดไป

    กรุณาตอบในรูปแบบนี้:
    "ระดับ ESI: [1-5] | คำแนะนำ: [ควรพบแพทย์ทันที / สามารถรอวันถัดไปได้]"
    """

    response = qa_pipeline(prompt, max_length=MODEL_MAX_LENGTH, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # ✅ ดึงค่า ESI Level ออกจากผลลัพธ์
    esi_level = 3  # ค่า Default ถ้าหาไม่เจอ
    if "ESI 1" in response:
        esi_level = 1
    elif "ESI 2" in response:
        esi_level = 2
    elif "ESI 3" in response:
        esi_level = 3
    elif "ESI 4" in response:
        esi_level = 4
    elif "ESI 5" in response:
        esi_level = 5

    return esi_level, response

