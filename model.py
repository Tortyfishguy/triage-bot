from transformers import pipeline

# โหลดโมเดล Perceptor AI ที่สามารถทำ Text Generation
MODEL_NAME = "Perceptor-AI/perceptor-medical-qa"
qa_pipeline = pipeline("text-generation", model=MODEL_NAME)

# ฟังก์ชันวิเคราะห์อาการทางการแพทย์ และจัดระดับ ESI (1-5)
def classify_esi(text):
    prompt = (
        f"ประเมินระดับความรุนแรงของอาการของผู้ป่วย: {text}\n"
        "ให้จัดระดับ ESI (Emergency Severity Index) ตั้งแต่ 1-5 โดยมีความหมายดังนี้:\n"
        "1 - อันตรายถึงชีวิต ต้องได้รับการรักษาทันที 🚨\n"
        "2 - อาการรุนแรง ต้องพบแพทย์โดยเร็ว ⏳\n"
        "3 - อาการปานกลาง รอการประเมินโดยแพทย์ได้ 🩺\n"
        "4 - อาการไม่รุนแรง สามารถรอรับบริการ OPD ได้ 📅\n"
        "5 - อาการเล็กน้อยมาก สามารถดูแลตนเองได้ที่บ้าน 🏡\n"
        "ตอบกลับเป็นข้อความที่ระบุระดับ ESI และคำแนะนำ"
    )

    response = qa_pipeline(prompt)
    return response[0]["generated_text"]
