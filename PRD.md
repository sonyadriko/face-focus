Problem Definition

Input:

Foto manusia
Seharusnya hanya ada 1 subjek utama
Kadang terdeteksi >1 wajah karena:
Poster di belakang
Foto di dinding
Pantulan kaca
Orang lewat di background

Output:

Foto hasil edit dengan hanya menyisakan wajah/subjek utama
Background tetap natural
Arsitektur V1 (Rule-Based + AI)
Image Upload
      ↓
Face Detection
      ↓
Face Classification
(Main Subject vs Background Face)
      ↓
Face Removal Mask Generation
      ↓
Inpainting
      ↓
Edited Image
Step 1. Face Detection

Model:

YOLOv11 Face
RetinaFace
InsightFace

Output:

[
  {
    "bbox": [100,200,300,400],
    "confidence": 0.99
  },
  {
    "bbox": [800,100,850,150],
    "confidence": 0.91
  }
]
Step 2. Main Subject Selection

Cari wajah utama berdasarkan skor:

score =
(face size × 50%)
+
(distance to center × 30%)
+
(face quality × 20%)

Misal:

Face	Size	Center	Score
User	besar	tengah	95
Poster	kecil	pojok	20

Maka:

main_face = user
remove_faces = [poster]
Step 3. Generate Removal Mask

Gunakan:

SAM2 (Segment Anything Model)
Grounded SAM

Hasil:

Poster Face Area
██████
██████

Mask ini yang akan dihapus.

Step 4. Inpainting

Pilihan model:

Open Source
Flux Fill
Stable Diffusion Inpainting
LaMa
Production Quality
OpenAI Image Editing
Flux Kontext
Ideogram Edit

Prompt:

Remove background face and reconstruct surrounding poster naturally.
V2 (Lebih Pintar)

Kalau ingin lebih akurat:

Background Face Classifier

Train model untuk klasifikasi:

MAIN_PERSON
POSTER_FACE
PHOTO_FRAME_FACE
REFLECTION_FACE
BACKGROUND_PERSON

Dataset bisa dibuat dari foto-foto operasionalmu.

Model:

EfficientNet
ConvNeXt
MobileNetV3

Karena sebenarnya "multiple face" tidak selalu berarti harus dihapus.

Contoh:

Foto keluarga = 4 wajah
→ jangan dihapus

Selfie = 1 wajah + poster
→ hapus poster
PRD Sederhana
Objective

Menghilangkan wajah non-subjek pada foto secara otomatis.

Success Metric
Metric	Target
Face Detection Accuracy	>95%
Main Subject Detection	>90%
Successful Removal	>85%
Processing Time	<5 detik
User Flow
Upload Foto
     ↓
Detect Faces
     ↓
Identify Main Subject
     ↓
Detect Background Faces
     ↓
Auto Remove
     ↓
Preview
     ↓
Download
Tech Stack yang Saya Rekomendasikan

Jika ingin cepat jadi:

Frontend
- Next.js

Backend
- Python FastAPI

Vision
- InsightFace
- SAM2

Editing
- Flux Fill

Storage
- S3 / MinIO

Queue
- Redis + Celery

Kalau targetnya seperti contoh foto yang tadi Anda kirim (wajah poster di belakang terdeteksi sebagai multiple face), saya malah akan membuat solusi yang lebih spesifik:

Face Detection
↓
Cari wajah terbesar (main face)
↓
Semua wajah lain dengan ukuran <30% wajah utama
↓
Auto remove menggunakan Flux Fill

Pendekatan ini biasanya sudah bisa menyelesaikan sekitar 80–90% kasus "multiple face karena poster/background" tanpa perlu training model khusus.