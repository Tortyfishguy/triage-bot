import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# เลือกโมเดล DeepSeek ที่จะใช้
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"

# โหลด Tokenizer และ Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def classify_esi(symptoms):
    """
    รับข้อความที่เป็นอาการของผู้ป่วยและใช้ DeepSeek AI วิเคราะห์ระดับ ESI
    """
    prompt = (
        f"คุณเป็นแพทย์ในห้องฉุกเฉินที่ต้องประเมินระดับ Emergency Severity Index (ESI) "
        f"ตามข้อมูลอาการของผู้ป่วย กรุณาประเมินและตอบกลับเฉพาะตัวเลข 1-5 เท่านั้น\n\n"
        f"อาการ: {symptoms}\n\n"
        f"ระดับ ESI:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        esi_level = int(response[-1])  # ดึงค่าตัวเลขสุดท้ายที่เป็นระดับ ESI
        return esi_level if 1 <= esi_level <= 5 else "ไม่สามารถประเมินได้"
    except ValueError:
        return "ไม่สามารถประเมินได้"
