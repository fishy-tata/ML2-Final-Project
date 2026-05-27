import io
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator
import xgboost
import joblib
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Dhanush" / "ARCFACE_MODEL_FINAL.joblib"


class BMIPredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BMIPrediction:
    bmi: float
    face_count: int
    model_path: str
    latency_ms: int


def _resolve_model_path() -> Path:
    configured = os.getenv("BMI_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_MODEL_PATH


def _json_event(event: str, **data: object) -> str:
    return json.dumps({"event": event, **data}) + "\n"


class ArcFaceBMIPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        if not self.model_path.exists():
            raise BMIPredictionError(
                f"Model file not found: {self.model_path}. "
                "Run `python scripts/train_arcface_bmi.py` first, or set BMI_MODEL_PATH."
            )

        self.model = joblib.load(self.model_path)
        expected_features = int(getattr(self.model, "n_features_in_", 512) or 512)
        if expected_features != 512:
            raise BMIPredictionError(
                f"Loaded model expects {expected_features} features. "
                "This API expects Dhanush's 512-d ArcFace model."
            )

        self.face_app = self._load_face_app()

    @staticmethod
    def _load_face_app():
        try:
            import insightface
        except ImportError as exc:
            raise BMIPredictionError(
                "Missing dependency `insightface`. Install dependencies from "
                "`bmi_api/requirements.txt` and restart the backend."
            ) from exc

        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(224, 224))
        return app

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise BMIPredictionError("Could not decode uploaded image.") from exc
        return np.asarray(image)

    def _extract_embedding(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        faces = self.face_app.get(image_rgb)
        if not faces:
            raise BMIPredictionError("No face was detected in the image.")

        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
        )
        embedding = np.asarray(largest_face.embedding, dtype=np.float32).reshape(1, -1)
        return embedding, len(faces)

    def predict(self, image_bytes: bytes) -> BMIPrediction:
        started_at = time.perf_counter()
        image_rgb = self._decode_image(image_bytes)
        embedding, face_count = self._extract_embedding(image_rgb)
        bmi = float(self.model.predict(embedding)[0])
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return BMIPrediction(
            bmi=round(bmi, 2),
            face_count=face_count,
            model_path=str(self.model_path),
            latency_ms=latency_ms,
        )


@lru_cache(maxsize=1)
def get_predictor() -> ArcFaceBMIPredictor:
    return ArcFaceBMIPredictor(_resolve_model_path())


def predict_bmi(image_bytes: bytes) -> BMIPrediction:
    return get_predictor().predict(image_bytes)


def stream_bmi_prediction(image_bytes: bytes) -> Iterator[str]:
    try:
        yield _json_event("status", message="Loading model")
        predictor = get_predictor()
        yield _json_event("status", message="Running face detection")
        result = predictor.predict(image_bytes)
        yield _json_event(
            "result",
            bmi=result.bmi,
            face_count=result.face_count,
            model_path=result.model_path,
            latency_ms=result.latency_ms,
        )
    except BMIPredictionError as exc:
        yield _json_event("error", message=str(exc))
    except Exception as exc:
        yield _json_event("error", message=f"Prediction failed: {exc}")
