import json
import os

import requests
import streamlit as st

try:
    from streamlit_theme import st_theme
except ImportError:
    st_theme = None


BACKEND_URL = os.getenv("BMI_BACKEND_URL", "http://localhost:8000")
CSS_PATH = os.path.join(os.path.dirname(__file__), "index.css")


def inject_styles() -> None:
    theme = st_theme() if st_theme else None
    is_dark = bool(theme and theme.get("textColor", "").lower() in {"#fff", "#ffffff"})
    if is_dark:
        theme_vars = """
            :root {
                --uchicago-maroon: #a4343a;
                --uchicago-maroon-hover: #8b1f25;
                --uchicago-text: #f7f5f0;
                --uchicago-border: #3c3630;
                --uchicago-muted: #c9c0b7;
                --uchicago-bg: #111111;
                --uchicago-panel: #1b1a18;
                --uchicago-uploader: #171614;
            }
        """
    else:
        theme_vars = """
            :root {
                --uchicago-maroon: #800000;
                --uchicago-maroon-hover: #5f0000;
                --uchicago-text: #171717;
                --uchicago-border: #ded9d2;
                --uchicago-muted: #6d6259;
                --uchicago-bg: #f7f5f0;
                --uchicago-panel: #ffffff;
                --uchicago-uploader: #ffffff;
            }
        """
    with open(CSS_PATH, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{theme_vars}</style>", unsafe_allow_html=True)
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def render_top_bar() -> None:
    st.markdown(
        """
        <div class="uchicago-top-bar">
            <a class="uchicago-brand" href="https://www.uchicago.edu/" target="_blank">
                The University of Chicago
            </a>
            <div class="uchicago-app-label">BMI from Facial Image</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="bmi-hero">
            <div class="bmi-kicker">Machine Learning Final Project</div>
            <h1 class="bmi-title">Real-time BMI Prediction</h1>
            <div class="bmi-subtitle">
                Upload a face image or use the camera. The API extracts an ArcFace embedding
                and reuses the loaded SVR model for low-latency BMI inference.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def stream_bmi_prediction(image_file):
    response = requests.post(
        f"{BACKEND_URL}/api/bmi/predict/stream",
        files={
            "image": (
                image_file.name,
                image_file.getvalue(),
                image_file.type or "image/png",
            )
        },
        stream=True,
        timeout=240,
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if line:
            yield json.loads(line)


def render_predictor() -> None:
    st.set_page_config(page_title="BMI Predictor", layout="centered")
    inject_styles()
    render_top_bar()
    render_header()

    input_col, result_col = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")

    with input_col:
        source = st.segmented_control(
            "Image source",
            ["Upload", "Camera"],
            default="Upload",
            label_visibility="collapsed",
        )

        image_file = None
        if source == "Upload":
            image_file = st.file_uploader("Face image", type=["png", "jpg", "jpeg", "webp"])
        else:
            image_file = st.camera_input("Take a face photo")

        if image_file is not None:
            st.image(image_file, use_container_width=True)

        predict_clicked = st.button(
            "Predict BMI",
            type="primary",
            disabled=image_file is None,
            use_container_width=True,
        )

    with result_col:
        status = st.empty()
        result = st.empty()
        details = st.empty()

    if not predict_clicked:
        with result_col:
            status.info("Select an image to run BMI inference.")
        return

    try:
        for event in stream_bmi_prediction(image_file):
            if event["event"] == "status":
                status.info(event["message"])
            elif event["event"] == "result":
                status.success("Prediction complete")
                result.metric("Predicted BMI", f"{event['bmi']:.2f}")
                details.markdown(
                    f"""
                    <div class="result-meta">
                    Faces detected: {event["face_count"]}<br>
                    Latency: {event["latency_ms"]} ms<br>
                    Model: {event["model_path"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif event["event"] == "error":
                status.error(event["message"])
    except requests.RequestException as exc:
        status.error(f"Backend request failed: {exc}")


if __name__ == "__main__":
    render_predictor()
