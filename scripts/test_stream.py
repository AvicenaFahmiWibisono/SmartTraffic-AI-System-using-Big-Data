
import cv2
import time
import os

urls = [
    ("Simpang Gedebage", "https://pelindung.bandung.go.id:3443/video/HIKSVISION/Soekarnogedebage.m3u8"),
    ("Pertigaan Waas", "https://pelindung.bandung.go.id:3443/video/HIKSVISION/WaaspertigaankompleksBatununggal.m3u8"),
    ("Kota Bogor Juanda", "https://restreamer3.kotabogor.go.id/memfs/e2d12ced-bcc3-4826-b872-97fcce335e93.m3u8")
]

# Set timeout for FFmpeg
# Try without custom user agent first, but longer timeout
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;20000"

print("Testing streams with OpenCV (Timeout 20s)...")
for name, url in urls:
    print(f"\nTesting {name}...")
    start = time.time()
    try:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS: {name} (Time: {time.time()-start:.2f}s)")
                print(f"Resolution: {frame.shape}")
            else:
                print(f"FAILED: {name} (Opened but no frame)")
        else:
            print(f"FAILED: {name} (Could not open)")
        cap.release()
    except Exception as e:
        print(f"ERROR: {name} - {e}")
