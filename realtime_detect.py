#  ''' .\venv\Scripts\Activate.ps1     '''

import cv2
import torch
import pandas as pd
import os
from datetime import datetime
import easyocr
from ultralytics import YOLO

# Load models
vehicle_model = YOLO("yolov8n.pt")  # Motorcycle detection (YOLOv5 class 3)
helmet_model = YOLO('runs/detect/train11/weights/best.pt')  # YOLOv8 helmet detection model

# EasyOCR for number plate reading
reader = easyocr.Reader(['en'])

# Output setup
os.makedirs('violations', exist_ok=True)
csv_path = 'violations/violations_log.csv'
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Timestamp", "Class", "Confidence", "PlateText", "ImagePath"]).to_csv(csv_path, index=False)

# Use laptop webcam or video file
cap = cv2.VideoCapture('helmet_test02.mp4')

# Optional: Set webcam resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
process_every_n_frames = 3

# Motorcycle detection
def detect_motorcycles(frame):
    results = vehicle_model(frame)
    for r in results:
        detections = r.boxes
        output = []
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 3 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                output.append([x1, y1, x2, y2, conf, cls_id])
        return output
    return []

# Helmet detection
def detect_helmet(crop):
    results = helmet_model.predict(source=crop, conf=0.4, imgsz=320)
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = helmet_model.names[cls_id]
            if label in ["helmet", "no-helmet"]:
                return label, conf
    return None, None

# Plate recognition
def recognize_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    return result[0][1] if result else "Unclear"

# Violation logging
def log_violation(cls_label, conf, plate, image):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"violations/{timestamp}_{plate.replace(' ', '')}.jpg"
    cv2.imwrite(filename, image)
    new_row = pd.DataFrame([[timestamp, cls_label, f"{conf:.2f}", plate, filename]],
                           columns=["Timestamp", "Class", "Confidence", "PlateText", "ImagePath"])
    new_row.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"[Violation] ❌ {cls_label} | Plate: {plate} | Saved: {filename}")

# Detection loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue

        motorcycles = detect_motorcycles(frame)

        for moto in motorcycles:
            x1, y1, x2, y2 = map(int, moto[:4])
            crop = frame[y1:y2, x1:x2]

            label, conf = detect_helmet(crop)

            if label == "no-helmet" and conf > 0.5:
                plate_crop = crop[-60:, :]
                plate = recognize_plate(plate_crop)
                log_violation(label, conf, plate, frame)

            # Draw
            color = (0, 0, 255) if label == "no-helmet" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Set window always on top to avoid taskbar overlay
        cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Helmet Detection", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")
