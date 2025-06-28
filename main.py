import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Load YOLOv11 player detection model ===
model = YOLO(r"C:\Users\Kashish Gupta\python\Soccer_Player_Identification\best.pt")

# === Initialize DeepSORT with higher max_age for re-identification ===
tracker = DeepSort(
    max_age=150,            # Allow players to disappear for ~5 sec at 30 FPS
    n_init=3,
    nms_max_overlap=1.0
)

# === Open input video ===
video_path = "15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Output video writer ===
out = cv2.VideoWriter(
    "output_tracked.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

print("🚀 Processing started...")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

    # === YOLOv11 detection ===
    results = model(frame, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # === Prepare DeepSORT detections ===
    formatted_detections = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det
        conf = confidences[i]
        cls_id = int(class_ids[i])

        print(f"Frame {frame_count}: Class {cls_id}, Conf {conf:.2f}, Box [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        # Filter for player classes (0, 2) with confidence threshold
        if cls_id in [0, 2] and conf > 0.3:
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                # Optional: tighter crop for better ReID stability
                x1 += w * 0.05
                y1 += h * 0.05
                w *= 0.9
                h *= 0.9

                formatted_detections.append(([x1, y1, w, h], conf, 'player'))

    # === DeepSORT tracking ===
    tracks = tracker.update_tracks(formatted_detections, frame=frame)

    # === Draw tracked players ===
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        color = (
            (track_id * 5) % 255,
            (track_id * 3) % 255,
            (track_id * 7) % 255
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Player {track_id}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write frame to output
    out.write(frame)

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete. Output saved as output_tracked.mp4.")
