from fastapi import FastAPI
import os

# Bring in the same services modules you currently use for processing
from app.services import image_processing, measurement, recommendation

u2net_model = None
mediapipe_pose = None

app = FastAPI(title="Sizing Worker")

@app.on_event("startup")
def load_models():
    global u2net_model, mediapipe_pose

    # load your U2Net and MediaPipe here
    try:
        from app.utils.libs.networks import U2NET
        import mediapipe as mp

        u2net_model = U2NET("u2net")
        mediapipe_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    except Exception as e:
        # log or handle error
        u2net_model, mediapipe_pose = None, None

@app.post("/process-images")
def process_images(payload: dict):
    # Example: reuse your existing service code
    return image_processing.process_images(
        payload, u2net_model, mediapipe_pose
    )

@app.get("/health")
def health():
    return {
        "status": "worker healthy",
        "u2net_loaded": u2net_model is not None,
        "mediapipe_loaded": mediapipe_pose is not None
    }

