import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data_augmented_balanced.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "BMI" / "BMI" / "Data" / "Images"
DEFAULT_OUTPUT = PROJECT_ROOT / "Dhanush" / "ARCFACE_MODEL_FINAL.joblib"


def find_candidate_files(name: str) -> list[Path]:
    return sorted(PROJECT_ROOT.rglob(name))


def find_candidate_image_dirs() -> list[Path]:
    candidates = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_dir() and path.name.lower() == "images":
            candidates.append(path)
    return sorted(candidates)


def format_candidates(paths: list[Path]) -> str:
    if not paths:
        return "  none found"
    return "\n".join(f"  {path}" for path in paths[:10])


def load_face_app(ctx_id: int):
    import insightface

    providers = ["CPUExecutionProvider"] if ctx_id < 0 else None
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(224, 224))
    return app


def get_arcface_embedding(face_app, image_path: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = face_app.get(image_rgb)
    if not faces:
        return None

    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )
    return largest_face.embedding


def build_dataset(csv_path: Path, image_root: Path, ctx_id: int, limit: int | None):
    data = pd.read_csv(csv_path)
    data["full_path"] = data["name"].apply(lambda name: image_root / str(name))
    data = data[data["full_path"].apply(lambda path: Path(path).exists())].reset_index(drop=True)
    data["is_training"] = data["is_training"].astype(bool)

    if limit is not None:
        data = data.head(limit).copy()

    face_app = load_face_app(ctx_id)
    embeddings, targets, kept_rows, missed = [], [], [], []

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Extracting ArcFace embeddings"):
        embedding = get_arcface_embedding(face_app, Path(row["full_path"]))
        if embedding is None:
            missed.append(str(row["full_path"]))
            continue
        embeddings.append(embedding)
        targets.append(float(row["bmi"]))
        kept_rows.append(idx)

    if not embeddings:
        raise RuntimeError("No face embeddings were extracted. Check image paths and image quality.")

    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    filtered = data.loc[kept_rows].reset_index(drop=True)
    return x, y, filtered, missed


def train_model(x_train: np.ndarray, y_train: np.ndarray):
    grid = GridSearchCV(
        SVR(kernel="rbf"),
        {
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.5],
            "gamma": ["scale", "auto"],
        },
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    return grid.best_estimator_, grid.best_params_


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    if len(y_test) > 1:
        metrics["pearson_r"] = float(pearsonr(y_test, y_pred)[0])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save Dhanush ArcFace BMI model.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ctx-id", type=int, default=-1, help="-1 for CPU, 0 for first GPU")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
    args = parser.parse_args()

    if not args.csv.exists():
        candidates = find_candidate_files("data_augmented_balanced.csv")
        raise FileNotFoundError(
            "CSV not found.\n"
            f"Expected: {args.csv}\n"
            "Fix: place `data_augmented_balanced.csv` in the project root, or run:\n"
            "  python scripts/train_arcface_bmi.py --csv /path/to/data_augmented_balanced.csv "
            "--image-root /path/to/Images\n"
            "Candidate CSVs found:\n"
            f"{format_candidates(candidates)}"
        )
    if not args.image_root.exists():
        candidates = find_candidate_image_dirs()
        raise FileNotFoundError(
            "Image directory not found.\n"
            f"Expected: {args.image_root}\n"
            "Fix: put images under `BMI/BMI/Data/Images`, or run:\n"
            "  python scripts/train_arcface_bmi.py --csv /path/to/data_augmented_balanced.csv "
            "--image-root /path/to/Images\n"
            "Candidate image directories found:\n"
            f"{format_candidates(candidates)}"
        )

    x, y, data, missed = build_dataset(args.csv, args.image_root, args.ctx_id, args.limit)
    is_train = data["is_training"].to_numpy(dtype=bool)
    x_train, x_test = x[is_train], x[~is_train]
    y_train, y_test = y[is_train], y[~is_train]

    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Train/test split is empty. Check the `is_training` column.")

    model, best_params = train_model(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_path": str(args.output),
                "csv": str(args.csv),
                "image_root": str(args.image_root),
                "n_total_with_faces": int(len(x)),
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "missing_or_no_face": int(len(missed)),
                "best_params": best_params,
                "metrics": metrics,
            },
            indent=2,
        )
    )

    print(f"Saved model: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
