import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# folders
input_folder = "."
output_folder = "final_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Βρίσκουμε όλα τα mp4 αρχεία
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

# Φόρτωση YOLOv8 Medium (για να πιάνει και την μπάλα)
model = YOLO("yolov8m.pt")


def run_tracking(video_path, mode_name, use_deep=True):
    """Εκτελεί tracking και αποθηκεύει βίντεο + TXT αποτελέσματα"""
    cap = cv2.VideoCapture(video_path)
    base_name = os.path.basename(video_path).split('.')[0]
    out_video_path = os.path.join(output_folder, f"{base_name}_{mode_name}.mp4")
    out_txt_path = os.path.join(output_folder, f"{base_name}_{mode_name}_results.txt")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    txt_file = open(out_txt_path, "w")

    # Ρύθμιση Tracker: Αν use_deep=False, το κάνουμε να συμπεριφέρεται σαν απλό SORT
    tracker = DeepSort(
        max_age=30 if use_deep else 5,
        n_init=3,
        nms_max_overlap=0.5,
        max_cosine_distance=0.2 if use_deep else 0.9,  # 0.9 σημαίνει αγνοεί την εμφάνιση
        embedder="mobilenet",
        half=True,
        embedder_gpu=True
    )

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_idx += 1

        # Ανίχνευση (Πιάνουμε παίκτες, μπάλα και οχήματα)
        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 32: sports ball
        results = model(frame, verbose=False, conf=0.3, classes=[0, 2, 3, 5, 7, 32])[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], score, int(class_id)))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            ltrb = track.to_ltrb()

            # Εγγραφή σε MOT Format για Metrics
            txt_file.write(
                f"{frame_idx},{tid},{int(ltrb[0])},{int(ltrb[1])},{int(ltrb[2] - ltrb[0])},{int(ltrb[3] - ltrb[1])},1,-1,-1,-1\n")

            # Σχεδίαση
            color = (0, 255, 0) if use_deep else (0, 0, 255)  # Πράσινο για Deep, Κόκκινο για Baseline
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
            cv2.putText(frame, f"ID:{tid}", (int(ltrb[0]), int(ltrb[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_video.write(frame)
        if frame_idx % 50 == 0: print(f"Processing {base_name} ({mode_name}): Frame {frame_idx}")

    cap.release()
    out_video.release()
    txt_file.close()


# --- ΚΥΡΙΩΣ LOOP ---
for v_file in video_files:
    # 1. Τρέξε DeepSORT (Πρόταση)
    run_tracking(v_file, "DeepSORT", use_deep=True)
    # 2. Τρέξε SORT (Baseline για σύγκριση)
    run_tracking(v_file, "SORT_Baseline", use_deep=False)

print("✅ Όλα τα βίντεο επεξεργάστηκαν! Δες τον φάκελο 'final_results'.")