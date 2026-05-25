# BMI API Deployment

This folder contains a new minimal frontend/backend for the BMI project. It does not modify the existing files under `/Users/jiayingzhong/Desktop/Gen AI`.

## 1. Train and save the model

Put the BMI data in the same layout used by Dhanush's notebook:

```text
data_augmented_balanced.csv
BMI/BMI/Data/Images/<image files>
```

Then run:

```bash
pip install -r bmi_api/requirements.txt
python scripts/train_arcface_bmi.py
```

This saves:

```text
Dhanush/ARCFACE_MODEL_FINAL.joblib
Dhanush/ARCFACE_MODEL_FINAL.metadata.json
```

For a quick smoke test:

```bash
python scripts/train_arcface_bmi.py --limit 100
```

## 2. Run the backend

```bash
pip install -r bmi_api/requirements.txt
uvicorn app.main:app --app-dir bmi_api --reload --port 8000
```

The backend loads `Dhanush/ARCFACE_MODEL_FINAL.joblib` once and reuses it for later requests.

If the model is somewhere else:

```bash
BMI_MODEL_PATH=/path/to/ARCFACE_MODEL_FINAL.joblib uvicorn app.main:app --app-dir bmi_api --reload --port 8000
```

## 3. Run the frontend

```bash
pip install -r bmi_frontend/requirements.txt
streamlit run bmi_frontend/main.py
```

The Streamlit app calls:

```text
POST http://localhost:8000/api/bmi/predict/stream
```

The streaming response is newline-delimited JSON status events followed by the BMI result.
