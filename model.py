import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 🔹 ตั้งค่าชื่อโมเดลและโฟลเดอร์เก็บโมเดล
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
MODEL_DIR = "./deepseek_model"

# 🔹 ตรวจสอบและสร้างโฟลเดอร์เก็บโมเดลหากยังไม่มี
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 🔹 โหลด Tokenizer และ Model พร้อมใช้ offload เพื่อลด RAM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

# 🔹 ใช้ quantization 4-bit เพื่อลด RAM ที่ใช้
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=MODEL_DIR,  # ใช้ offload เพื่อลดการใช้ RAM
    quantization_config=quantization_config
)

def classify_esi(symptoms):
    """
    🔹 วิเคราะห์ระดับ ESI จากอาการของผู้ป่วยโดยใช้ DeepSeek AI
    """
    prompt = (
        "คุณเป็นแพทย์เวชศาสตร์ฉุกเฉินที่ต้องประเมินระดับความรุนแรงของผู้ป่วยตาม Emergency Severity Index (ESI)\n"
        "ESI แบ่งเป็น 5 ระดับ:\n"
        "- ESI 1: ภาวะฉุกเฉินวิกฤติ เช่น หัวใจหยุดเต้น, หยุดหายใจ, ความดันตกวิกฤติ\n"
        "- ESI 2: ภาวะเสี่ยงสูง เช่น สับสน, หมดสติ, หอบเหนื่อยรุนแรง\n"
        "- ESI 3: ต้องใช้ทรัพยากรทางการแพทย์หลายอย่าง เช่น ตรวจเลือด, X-ray, CT Scan\n"
        "- ESI 4: ต้องใช้ทรัพยากรทางการแพทย์เพียงอย่างเดียว เช่น เย็บแผล, ทำแผล\n"
        "- ESI 5: ไม่ต้องใช้ทรัพยากรทางการแพทย์ เช่น ไข้หวัดเล็กน้อย, ผื่นคัน\n\n"
        "อาการของผู้ป่วย: "
        f"{symptoms}\n\n"
        "กรุณาประเมินระดับ ESI และตอบกลับเป็นตัวเลข 1-5 เท่านั้น หากข้อมูลไม่เพียงพอให้ตอบว่า 'ไม่สามารถประเมินได้'"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 🔹 ตรวจสอบผลลัพธ์ที่โมเดลตอบกลับ
    for esi in ["1", "2", "3", "4", "5"]:
        if esi in response:
            return int(esi)

    return "ไม่สามารถประเมินได้"  # กรณีที่โมเดลไม่สามารถให้คำตอบที่เป็นตัวเลขได้
