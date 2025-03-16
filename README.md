# ESI Triage Bot
บอทคัดกรองอาการผู้ป่วยผ่าน LINE และประเมินระดับ ESI โดยใช้ WangchanBERTa

## 🔹 วิธีรันเซิร์ฟเวอร์
pip install -r requirements.txt python app.py

## 🔹 วิธี Deploy บน Render
- เชื่อม GitHub → Deploy อัตโนมัติ
- ตั้งค่า Environment Variables:
  - `LINE_ACCESS_TOKEN`
  - `LINE_CHANNEL_SECRET`
  - `FIREBASE_CREDENTIALS_JSON`
