import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# เลือกโมเดล DeepSeek ที่จะใช้
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
MODEL_DIR = "./deepseek_model"

# ตรวจสอบและสร้างโฟลเดอร์เก็บโมเดลหากยังไม่มี
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# โหลด Tokenizer และ Model พร้อมใช้ offload เพื่อลดการใช้ RAM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=MODEL_DIR  # ใช้ offload เพื่อลดการใช้ RAM
)

def classify_esi(symptoms):
    """
    รับข้อความที่เป็นอาการของผู้ป่วยและใช้ DeepSeek AI วิเคราะห์ระดับ ESI
    """
    prompt = (
        "คุณเป็นแพทย์เวชศาสตร์ฉุกเฉินที่มีหน้าที่ประเมินระดับความรุนแรงของผู้ป่วยตาม Emergency Severity Index (ESI) ซึ่งแบ่งเป็น 5 ระดับ:\n"
        "- ESI 1: ต้องได้รับการช่วยชีวิตทันที เช่น หัวใจหยุดเต้น หยุดหายใจ ความดันต่ำวิกฤติ\n"
        "- ESI 2: มีภาวะเสี่ยงสูง เช่น สับสน หมดสติ อาการกำเริบที่อาจรุนแรง\n"
        "- ESI 3: ต้องใช้ทรัพยากรทางการแพทย์หลายอย่าง เช่น ตรวจเลือดและเอกซเรย์\n"
        "- ESI 4: ต้องใช้ทรัพยากรทางการแพทย์เพียงอย่างเดียว เช่น ทำแผล เย็บแผล\n"
        "- ESI 5: ไม่ต้องใช้ทรัพยากรทางการแพทย์ เช่น เป็นหวัดเล็กน้อย แผลถลอก\n\n"
        "อาการของผู้ป่วย: "
        f"{symptoms}\n\n"
        "กรุณาประเมินระดับ ESI ที่เหมาะสมและตอบกลับเป็นตัวเลข 1-5 เท่านั้น หากข้อมูลไม่เพียงพอให้ตอบว่า 'ไม่สามารถประเมินได้'"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ตรวจสอบการตอบกลับของโมเดล
    for esi in ["1", "2", "3", "4", "5"]:
        if esi in response:
            return int(esi)

    return "ไม่สามารถประเมินได้"  # กรณีที่โมเดลไม่สามารถให้คำตอบที่เป็นตัวเลขได้
