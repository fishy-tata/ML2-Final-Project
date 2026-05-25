from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.bmi_model import BMIPredictionError, predict_bmi, stream_bmi_prediction


app = FastAPI(title="BMI Prediction API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/bmi/predict")
async def bmi_predict(image: UploadFile = File(...)) -> dict[str, object]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    try:
        result = predict_bmi(image_bytes)
    except BMIPredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "bmi": result.bmi,
        "face_count": result.face_count,
        "model_path": result.model_path,
        "latency_ms": result.latency_ms,
    }


@app.post("/api/bmi/predict/stream")
async def bmi_predict_stream(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    return StreamingResponse(
        stream_bmi_prediction(image_bytes),
        media_type="application/x-ndjson",
    )
