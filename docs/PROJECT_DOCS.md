# Dokumentasi Pembaruan Sistem Vehicle Counter

Dokumen ini merangkum fitur-fitur baru dan pembaruan yang telah diterapkan pada sistem Vehicle Counter.

## 1. Fitur Baru: Peta Dashboard Interaktif (Interactive Dashboard Map)
**Lokasi:** Halaman Dashboard (`/dashboard`)
**Deskripsi:**
Menambahkan visualisasi peta interaktif menggunakan **Leaflet.js** untuk memantau lokasi kamera CCTV.
*   **Peta Pulau Jawa:** Default view peta berpusat di Pulau Jawa.
*   **Marker Kamera:** Setiap kamera yang memiliki koordinat (Latitude & Longitude) ditampilkan sebagai titik di peta.
*   **Indikator Kemacetan (Congestion Indicator):** Warna marker berubah secara real-time berdasarkan kepadatan lalu lintas saat ini (*Current Density*).
    *   🟢 **HIJAU (SEPI):** ≤ 15 kendaraan.
    *   🟡 **KUNING (SEDANG):** 16 - 35 kendaraan.
    *   🔴 **MERAH (RAMAI):** > 35 kendaraan.
*   **Detail Popup:** Klik marker untuk melihat:
    *   Nama Kamera
    *   Status Kepadatan (SEPI/SEDANG/RAMAI)
    *   **Current Density:** Jumlah kendaraan saat ini (Real-time).
    *   **Rincian Kendaraan:** Jumlah Mobil (🚗) dan Motor (🏍️) saat ini.
    *   **Info Cuaca:** Menampilkan cuaca, suhu, dan kecepatan angin real-time di lokasi CCTV.
    *   **Total Akumulasi:** Total kendaraan yang telah lewat sejak sistem berjalan.

## 2. Fitur Baru: Manajemen Koordinat Kamera
**Lokasi:** Halaman Utama (`/`) & API
**Deskripsi:**
Memungkinkan pengguna untuk menambahkan dan mengedit lokasi geografis kamera.
*   **Input Koordinat:** Form Tambah Kamera dan Edit Kamera kini memiliki field **Latitude** dan **Longitude**.
*   **Tombol Edit:** Menambahkan tombol Edit (ikon pensil) pada daftar kamera untuk mengubah nama, URL, dan koordinat tanpa menghapus kamera.
*   **API Endpoints:**
    *   `POST /api/edit_camera`: Endpoint baru untuk menyimpan perubahan konfigurasi kamera.

## 3. Fitur Baru: Navigasi "View Map"
**Lokasi:** Halaman Dashboard - Kartu Kamera
**Deskripsi:**
Tombol cepat untuk melihat lokasi kamera di peta.
*   Menambahkan tombol **Map** di sebelah tombol Monitor pada setiap kartu kamera.
*   **Fungsi:** Ketika diklik, halaman akan scroll ke peta dan melakukan animasi zoom (*fly-to*) ke lokasi kamera tersebut.

## 4. Pembaruan Sistem (System Updates)
*   **Watermark Video:** Menambahkan watermark teks "desavitho" pada setiap frame video yang diproses oleh AI (file: `app/services/camera.py`).
*   **Perbaikan Git:** Konfigurasi autentikasi SSH untuk repositori GitHub `desavitho/big-data-traffict-counting-with-yolo`.
*   **Layanan Systemd:** Verifikasi dan restart service `vehicle-counter.service` untuk penerapan perubahan.

## 5. Rincian Teknis
*   **Frontend Library:** Menambahkan `Leaflet.js` (CSS & JS) pada `dashboard.html`.
*   **Logic Kepadatan:** Menggunakan data `current_count` dari backend untuk visualisasi real-time di peta, menggantikan logika rata-rata 10 detik sebelumnya agar lebih akurat dan responsif.
*   **Data Breakdown:** Popup peta kini menampilkan pemisahan data antara Mobil (Class 0/2) dan Motor (Class 1/3).

---
*Dibuat otomatis oleh AI Assistant - 2026-02-02*
