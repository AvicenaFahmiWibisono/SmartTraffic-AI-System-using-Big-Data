# 🚦 SmartTraffic AI System

> Platform Edge-Computing untuk Pemantauan Lalu Lintas Real-Time dan Prediksi Volume Kendaraan Berbasis Big Data

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://postgresql.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-red?logo=pytorch)](https://pytorch.org)

---

## 📋 Deskripsi

SmartTraffic AI mengubah **stream CCTV mentah** menjadi **insight numerik** melalui pipeline:

```
CCTV Stream → YOLOv8 Detection → IoU Tracking → PostgreSQL → Transformer Forecaster → Dashboard
```

Sistem mampu memproses **37 stream CCTV** secara paralel, menyimpan **455K+ data points** di PostgreSQL, dan memprediksi volume kendaraan **1–48 jam ke depan** menggunakan Tiny Transformer Encoder.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🎥 **Real-time Detection** | YOLOv8 mendeteksi mobil & motor dari 37 CCTV stream |
| 🧠 **Transformer Forecaster** | Prediksi volume kendaraan per jam (d_model=32, 2 layers, 4 heads) |
| 🗺️ **Interactive Map** | Peta Leaflet dengan routing berwarna gradien sesuai kongesti |
| 📊 **Analytics Dashboard** | Chart.js untuk tren historis, perbandingan kamera |
| 🤖 **AI Assistant** | Chatbot terintegrasi (Ollama LLM) yang bisa navigasi UI |
| 🐘 **PostgreSQL** | Database production-ready dengan real-time read/write |
| 🔄 **Live Sync** | Edit data di pgAdmin → dashboard langsung berubah |
| 📡 **36 Kamera** | Monitoring titik-titik strategis di Bandung & Bogor |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SmartTraffic AI System                         │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│  Acquire │  Detect  │  Persist │   AI/ML  │   API    │   Client     │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ CCTV     │ YOLOv8   │PostgreSQL│Transformer│ Flask   │ Leaflet Map  │
│ RTSP/HLS │ IoU Track│ Data Lake│DOW×Hour  │ REST    │ Chart.js     │
│ FFmpeg   │ NMS      │ JSON snap│ Ollama   │ /api/*  │ AI Chatbot   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- (Opsional) Ollama untuk AI Assistant

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AvicenaFahmiWibisono/SmartTraffic-AI-System-using-Big-Data.git
cd SmartTraffic-AI-System-using-Big-Data

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup PostgreSQL
#    Buat database "smarttraffic" di PostgreSQL server
#    Tabel otomatis dibuat saat server pertama kali start

# 4. Set environment variables
set DB_PASSWORD=your_postgres_password
# atau di Linux/Mac:
# export DB_PASSWORD=your_postgres_password

# 5. (Opsional) Setup Ollama untuk AI chatbot
ollama pull mistral:7b

# 6. Jalankan server
python run.py
```

Server berjalan di **http://localhost:5000**

### Halaman yang Tersedia

| URL | Fungsi |
|-----|--------|
| `/` | Monitor live (video feed + deteksi) |
| `/dashboard` | Peta interaktif + statistik + routing |
| `/analysis` | Grafik tren & analitik historis |
| `/documentation` | Dokumentasi sistem lengkap |

---

## ⚙️ Konfigurasi

Semua konfigurasi via environment variables:

```env
# PostgreSQL (wajib)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=smarttraffic
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama AI (opsional)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

Lihat `.env.example` untuk template lengkap.

---

## 🧠 Transformer Forecaster

### Cara Kerja

1. **Input**: 48 jam terakhir volume kendaraan per kamera
2. **Encoding**: Value + Hour embedding (24) + DOW embedding (7) + Positional
3. **Processing**: 2-layer Transformer Encoder, 4 attention heads, d_model=32
4. **Output**: Prediksi volume 1 jam ke depan
5. **Multi-step**: Autoregressive sampai 48 jam ke depan

### Spesifikasi Model

| Parameter | Nilai |
|-----------|-------|
| d_model | 32 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Context window | 48 jam |
| Max forecast | 48 jam (autoregressive) |
| Training time | 2-5 detik (CPU) |
| Min data required | 96 hourly buckets (~4 hari) |

### Klasifikasi Kongesti

| Volume/jam | Status | Warna |
|-----------|--------|-------|
| < 500 | LANCAR | 🟢 Hijau |
| 500 – 1000 | PADAT LANCAR | 🟡 Kuning |
| 1000 – 2000 | MACET | 🟠 Oranye |
| > 2000 | MACET TOTAL | 🔴 Merah |

---

## 🐘 PostgreSQL Integration

### Tabel Utama

| Tabel | Fungsi |
|-------|--------|
| `traffic_history` | Data historis per interval per kamera (455K+ rows) |
| `live_status` | Current count per kamera (real-time, editable) |
| `chat_profile` | Profil session AI Assistant |
| `chat_messages` | Riwayat percakapan |

### Demo: Edit via pgAdmin

1. Klik **"DB: Writing"** di dashboard → berubah jadi **"DB: Paused"**
2. Edit di pgAdmin:
```sql
UPDATE live_status SET current_count = 300 WHERE camera_name = 'Simpang Braga';
```
3. Dashboard otomatis berubah (polling 10 detik)
4. Klik **"DB: Paused"** → kembali ke **"DB: Writing"**

### Query Prediksi

```sql
-- Prediksi volume untuk hari Rabu jam 8 pagi
WITH hourly_sums AS (
    SELECT DATE(to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') AS tgl,
           SUM(new_count) AS hourly_total
    FROM traffic_history
    WHERE camera_name = 'Simpang Braga'
      AND EXTRACT(DOW FROM to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') = 3
      AND EXTRACT(HOUR FROM to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') = 8
    GROUP BY tgl
)
SELECT AVG(hourly_total)::int AS prediksi,
       CASE WHEN AVG(hourly_total) > 1000 THEN 'MACET'
            WHEN AVG(hourly_total) > 500 THEN 'PADAT LANCAR'
            ELSE 'LANCAR' END AS status
FROM hourly_sums;
```

---

## 📁 Struktur Project

```
SmartTraffic-AI-System/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Konfigurasi (paths, thresholds)
│   ├── database.py          # PostgreSQL + Transformer model
│   ├── globals.py           # Shared state (global_stats, locks)
│   ├── routes.py            # API endpoints + chat logic
│   ├── utils.py             # Helper functions
│   ├── services/
│   │   └── camera.py        # Camera agents + YOLOv8 inference
│   ├── static/img/
│   │   └── logo.png
│   └── templates/
│       ├── base.html        # Layout + chatbot
│       ├── index.html       # Monitor page
│       ├── dashboard.html   # Map + stats + routing
│       ├── analysis.html    # Analytics charts
│       └── documentation.html # System docs
├── data/
│   ├── cctv_config.json     # Konfigurasi 36 kamera
│   ├── camera_thresholds.json
│   └── traffic_stats.json   # In-memory snapshot
├── models/
│   ├── yolov5n.onnx         # Fallback ONNX model
│   └── yolov8n.pt           # YOLOv8 nano
├── scripts/                  # Utility scripts
├── docs/                     # Dokumentasi tambahan
├── migrate_to_postgres.py    # Script migrasi SQLite → PostgreSQL
├── predict_fill.py           # Fill missing predictions
├── requirements.txt
├── run.py                    # Entry point
├── .env.example              # Template environment variables
└── .gitignore
```

---

## 🔌 API Reference

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | `/api/stats` | Statistik real-time semua kamera |
| GET | `/api/history` | Data historis (period: 30m/1h/6h/24h/7d/30d) |
| POST | `/api/predict_traffic` | Prediksi volume per kamera |
| POST | `/api/chat` | AI Assistant endpoint |
| GET/POST | `/api/db_write_pause` | Toggle pause/resume DB writes |
| GET | `/api/export_csv` | Export data ke CSV |
| POST | `/api/add_camera` | Tambah kamera baru (admin) |
| POST | `/api/edit_camera` | Edit kamera (admin) |
| POST | `/api/reset_data` | Reset semua data (admin) |

---

## 📊 Data & Big Data (4 Vs)

| V | Implementasi |
|---|-------------|
| **Volume** | 455K+ rows di PostgreSQL, partisi CSV di Data Lake |
| **Velocity** | 37 stream paralel, frame → DB < 500ms |
| **Variety** | RTSP/HLS dari berbagai vendor CCTV → JSON/CSV/Leaflet |
| **Value** | Prediksi kongesti, rekomendasi rute, laporan otomatis |

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Backend | Python 3.12, Flask 3.0 |
| AI/ML | YOLOv8 (Ultralytics), PyTorch Transformer, Ollama |
| Database | PostgreSQL 16, psycopg2 |
| Frontend | Tailwind CSS, Leaflet.js, Chart.js, Mermaid.js |
| Routing | OSRM (Open Source Routing Machine) |
| Video | OpenCV, FFmpeg |

---

## 👤 Author

**Avicena Fahmi Wibisono**

- Website: [avicenafahmi.com](https://avicenafahmi.com)
- GitHub: [@AvicenaFahmiWibisono](https://github.com/AvicenaFahmiWibisono)

---

## 📄 License

This project is for educational purposes — Big Data Traffic Prediction System.
