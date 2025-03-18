import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# โหลดโมเดล Mistral-7B-Instruct
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def classify_esi(patient_symptoms: str) -> str:
    """
    วิเคราะห์อาการของผู้ป่วยและจัดระดับ ESI (Emergency Severity Index)
    """
    prompt = f"""
    คุณเป็นแพทย์ฉุกเฉินที่เชี่ยวชาญด้านการคัดกรองผู้ป่วยฉุกเฉินโดยใช้ Emergency Severity Index (ESI)
    วิเคราะห์อาการต่อไปนี้และจัดระดับ ESI จาก 1 ถึง 5:
    
    อาการของผู้ป่วย: {patient_symptoms}

    คำตอบของคุณควรอยู่ในรูปแบบ:
    - ESI Level: (ระดับที่ 1 ถึง 5)
    - คำแนะนำ: (ควรพบแพทย์ทันทีหรือรอได้)

    คำตอบ:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
