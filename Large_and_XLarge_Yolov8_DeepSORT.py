import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---
#In order to run the extra large version of YOLOv8 you have to exchange these 3 rows with the one in comments
VIDEO_PATH = "videoplayback.mp4"
#OUTPUT_PATH = "final_stable_tracking_x_traffic.mp4"
OUTPUT_PATH = "final_stable_tracking_l_traffic.mp4"
#METRIC_FILE = "my_results_x_traffic.txt"
METRIC_FILE = "my_results_l_traffic.txt"

# 1. THE BRAIN: YOLOv8 Large
#model = YOLO("yolov8x.pt")
model = YOLO("yolov8l.pt")

# 2. THE MEMORY: Adjusted for stability
tracker = DeepSort(
    max_age=40,
    n_init=5,
    nms_max_overlap=0.3,
    max_cosine_distance=0.2,
    nn_budget=200
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

if os.path.exists(METRIC_FILE): os.remove(METRIC_FILE)

# General classes: Person, Bicycle, Car, Motorcycle, Bus, Truck
ALLOWED_CLASSES = [0, 1, 2, 3, 5, 7]

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # Stage 1: Detection
    # Using 1088 to satisfy the 'multiple of 32' warning and avoid errors
    results = model(frame, imgsz=1088, conf=0.45, classes=ALLOWED_CLASSES)[0]

    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        w, h = x2 - x1, y2 - y1
        detections.append(([int(x1), int(y1), int(w), int(h)], score, int(class_id)))

    # Stage 2: Association
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        # Check 1: Track must be confirmed (seen for n_init frames)
        if not track.is_confirmed():
            continue

        # Check 2: Track must have been updated in this specific frame
        # This prevents the "walking ghost boxes"
        if track.time_since_update > 1:
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        ltwh = track.to_ltwh()

        # Visuals
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Logging for MOTA
        with open(METRIC_FILE, "a") as f:
            f.write(f"{frame_count},{track_id},{ltwh[0]},{ltwh[1]},{ltwh[2]},{ltwh[3]},1,-1,-1,-1\n")

    cv2.imshow("Final Stable Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()