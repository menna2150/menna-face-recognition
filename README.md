# Face Recognition + Emotion Detection (Menna)

Real-time webcam app that detects a single enrolled person ("Menna"), labels everyone else as *Unknown*, and overlays the person's current emotion. Built as a Google Colab notebook using RetinaFace for face detection and DeepFace (FaceNet512) for identity + emotion analysis.


## Features
- Live webcam capture inside Google Colab (JS ↔ Python bridge).
- Face detection with **RetinaFace**.
- Identity matching with **DeepFace / FaceNet512** against an enrolled reference set.
- Emotion classification (happy, sad, angry, surprise, fear, disgust, neutral) with emoji labels.
- Two modes:
  - Detect emotion for **all** faces.
  - Detect emotion **only for the enrolled person**.
- Single-image test mode (upload a photo and see results).

## Tech stack
- Python, OpenCV, NumPy
## How to run
1. Open `menna_face_recognition.ipynb` in Google Colab.
2. **Cell 1** — install dependencies.
3. **Cell 2–3** — imports + load the RetinaFace detector.
4. **Cell 4** — upload 3–10 clear photos of the person to enroll (face visible, good lighting). They are saved under `/content/menna_db/Menna/`.
5. **Cell 5** — load helper functions (detection, recognition, emotion, drawing).
6. **Cell 6 / Cell 9** — start the live webcam stream. A green box marks the enrolled person; a red box marks unknown faces. Click **Stop Webcam** to end.
7. (Optional) **Cell 8** — run on a single uploaded image instead of the webcam.

## Project structure
```
.
├── menna_face_recognition.ipynb   # main notebook
└── README.md
```

## Notes
- Recognition threshold is the FaceNet512 distance `< 0.45`. Lower it for stricter matching or raise it to be more permissive.
- The notebook is built for **Google Colab** (uses `google.colab.files` and JS webcam capture); running locally would require swapping the capture cell for `cv2.VideoCapture(0)`.
- Re-enroll by re-running Cell 4 and clearing the cached embeddings (`representations_facenet512.pkl`).

