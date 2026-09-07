from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="Hiwet Breast Cancer AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 1.2rem; padding-bottom: 3rem;}
.hero {padding: 1.7rem 2rem; border: 1px solid rgba(128,128,128,.25); border-radius: 20px; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(0,100,200,.08), rgba(0,160,100,.05));}
.hero h1 {margin-bottom: .25rem;}
.hero p {font-size: 1.05rem; margin-bottom: 0;}
.section {padding: 1.1rem 1.2rem; border: 1px solid rgba(128,128,128,.22); border-radius: 16px; margin-bottom: 1rem;}
.small-muted {color: rgba(120,120,120,.95); font-size: .9rem;}
.result-card {padding: 1.4rem; border: 2px solid rgba(128,128,128,.25); border-radius: 18px; text-align: center;}
.badge {display:inline-block; padding:.25rem .65rem; border-radius:999px; border:1px solid rgba(128,128,128,.3); font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

@st.cache_data
def load_dataset():
    return load_breast_cancer()

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

model, scaler = load_assets()
data = load_dataset()
metrics = load_metrics()
features = list(data.feature_names)
dataset = pd.DataFrame(data.data, columns=features)

BENIGN_EXAMPLE = dataset.loc[data.target == 1].iloc[0].to_dict()
MALIGNANT_EXAMPLE = dataset.loc[data.target == 0].iloc[0].to_dict()
DEFAULT_EXAMPLE = dataset.median().to_dict()


def set_values(values):
    for feature in features:
        st.session_state[f"input_{feature}"] = float(values[feature])


def clean_uploaded_csv(df):
    """Accept common spreadsheet index columns, then require the exact 30 model features."""
    df = df.copy()
    removable = [c for c in df.columns if str(c).strip().lower() in {"index", "unnamed: 0"}]
    if removable:
        df = df.drop(columns=removable)
    return df


for feature in features:
    st.session_state.setdefault(f"input_{feature}", float(DEFAULT_EXAMPLE[feature]))
st.session_state.setdefault("case_id", "")

with st.sidebar:
    st.title("🩺 Hiwet Breast Cancer AI")
    st.caption("Clinical decision-support prototype")
    st.divider()

    st.markdown("### Quick test")
    if st.button("🟢 Load benign example", use_container_width=True):
        set_values(BENIGN_EXAMPLE)
        st.session_state["data_source"] = "Demo benign example"
        st.rerun()
    if st.button("🔴 Load malignant example", use_container_width=True):
        set_values(MALIGNANT_EXAMPLE)
        st.session_state["data_source"] = "Demo malignant example"
        st.rerun()
    if st.button("↩️ Reset", use_container_width=True):
        set_values(DEFAULT_EXAMPLE)
        st.session_state["data_source"] = "Manual entry"
        st.rerun()

    st.divider()
    st.markdown("### Model information")
    st.write("**Algorithm:** Support Vector Machine")
    st.write("**Kernel:** RBF")
    st.write("**Input features:** 30")
    auc = metrics.get("roc_auc")
    if auc is None:
        auc = metrics.get("test_metrics", {}).get("ROC-AUC")
    if auc is not None:
        st.write(f"**Test ROC-AUC:** {float(auc):.4f}")

st.markdown("""
<div class="hero">
<h1>🩺 Hiwet Breast Cancer AI</h1>
<p>Machine-learning decision-support prototype for classifying a diagnostic measurement record as benign or malignant.</p>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ Educational and research prototype only. This application does not provide a medical diagnosis and must not replace evaluation by a qualified healthcare professional.")

# Case information
st.markdown("## 1. Case information")
with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.text_input("Case ID (optional)", key="case_id", placeholder="e.g. HBC-2026-001")
    with c2:
        st.markdown("**What does the user provide?**")
        st.caption("The 30 numerical measurements should come from an appropriate diagnostic process or a validated data source. A patient should not try to measure these values themselves.")

# Data input
st.markdown("## 2. Diagnostic data")
tab_csv, tab_manual = st.tabs(["📄 Upload diagnostic CSV", "✍️ Manual entry"])

with tab_csv:
    st.markdown("**Upload the measurements produced by your diagnostic/data system.**")
    st.caption("The file must contain the 30 model feature names. A normal `index` or `Unnamed: 0` column is automatically ignored.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ CSV template", pd.DataFrame(columns=features).to_csv(index=False), "sample_template.csv", "text/csv", use_container_width=True)
    with c2:
        p = BASE_DIR / "sample_benign_features.csv"
        if p.exists():
            st.download_button("🟢 Benign demo CSV", p.read_bytes(), "sample_benign_features.csv", "text/csv", use_container_width=True)
    with c3:
        p = BASE_DIR / "sample_malignant_features.csv"
        if p.exists():
            st.download_button("🔴 Malignant demo CSV", p.read_bytes(), "sample_malignant_features.csv", "text/csv", use_container_width=True)

    uploaded = st.file_uploader("Choose diagnostic CSV", type=["csv"], key="diagnostic_csv")
    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            csv_df = clean_uploaded_csv(csv_df)
            missing = [f for f in features if f not in csv_df.columns]
            extra = [c for c in csv_df.columns if c not in features]

            if csv_df.empty:
                st.error("❌ The uploaded CSV has headers but no data rows. Please upload a CSV containing at least one record.")
            elif missing:
                st.error(f"❌ Missing {len(missing)} required feature(s): {', '.join(missing)}")
            elif extra:
                st.error(f"❌ Unexpected column(s): {', '.join(map(str, extra))}")
            else:
                ordered = csv_df[features].apply(pd.to_numeric, errors="raise")
                if not np.isfinite(ordered.to_numpy()).all():
                    raise ValueError("all 30 feature values must be finite numbers")
                set_values(ordered.iloc[0].to_dict())
                st.session_state["data_source"] = f"CSV: {uploaded.name}"
                st.success(f"✅ CSV validated: {len(ordered)} record(s), 30/30 required features found. First record loaded.")
                if len(ordered) > 1:
                    st.info(f"This demo uses the first of {len(ordered)} records. For batch prediction, add a batch-prediction workflow later.")
                with st.expander("Preview loaded diagnostic record"):
                    st.dataframe(ordered.head(1), use_container_width=True)
        except Exception as exc:
            st.error(f"❌ Could not process the CSV: {exc}")

with tab_manual:
    st.caption("Manual entry is mainly for controlled testing or demonstration. In a real workflow, measurements should come from an appropriate diagnostic/data source.")

st.markdown("## 3. Diagnostic measurements")
groups = {
    "Mean measurements": [f for f in features if f.startswith("mean ")],
    "Standard-error measurements": [f for f in features if f.endswith(" error")],
    "Worst measurements": [f for f in features if f.startswith("worst ")],
}
for title, group in groups.items():
    with st.container(border=True):
        st.markdown(f"### {title}")
        cols = st.columns(3)
        for i, feature in enumerate(group):
            with cols[i % 3]:
                st.number_input(feature.title(), key=f"input_{feature}", format="%.6f")

# Validation summary
st.markdown("## 4. Pre-analysis validation")
input_df = pd.DataFrame([[st.session_state[f"input_{f}"] for f in features]], columns=features)
v1, v2, v3, v4 = st.columns(4)
v1.metric("Required features", "30")
v2.metric("Features loaded", str(input_df.shape[1]))
v3.metric("Missing values", str(int(input_df.isna().sum().sum())))
v4.metric("Data status", "Ready" if np.isfinite(input_df.to_numpy()).all() else "Check")

st.markdown("## 5. AI analysis")
predict = st.button("🔍 Run AI classification", type="primary", use_container_width=True)

if predict:
    if not np.isfinite(input_df.to_numpy()).all():
        st.error("Every measurement must be a finite numeric value.")
        st.stop()
    try:
        scaled = scaler.transform(input_df)
        prediction = int(model.predict(scaled)[0])
        probabilities = model.predict_proba(scaled)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    malignant = float(probabilities[0]) * 100
    benign = float(probabilities[1]) * 100
    label = "MALIGNANT" if prediction == 0 else "BENIGN"
    higher = max(malignant, benign)
    case_id = st.session_state.get("case_id", "") or "Not provided"

    st.markdown("## 6. AI result")
    if label == "MALIGNANT":
        st.error(f"### ⚠️ MALIGNANT\nModel-estimated malignant probability: **{malignant:.2f}%**")
    else:
        st.success(f"### ✅ BENIGN\nModel-estimated benign probability: **{benign:.2f}%**")

    a, b, c = st.columns(3)
    a.metric("Case ID", case_id)
    b.metric("Predicted class", label)
    c.metric("Higher estimated probability", f"{higher:.2f}%")

    result_df = pd.DataFrame({"Class": ["Malignant", "Benign"], "Estimated probability (%)": [malignant, benign]})
    st.bar_chart(result_df.set_index("Class"), height=300)

    st.info("Clinical interpretation: this output is a machine-learning classification estimate. A qualified healthcare professional must review the underlying diagnostic evidence and make any clinical decision.")

st.markdown("## 7. Ultrasound computer-vision analysis")
st.caption(
    "Upload a breast ultrasound image. The CNN is designed for the BUSI "
    "three-class dataset: benign, malignant, and normal."
)

CV_DIR = BASE_DIR / "cv"
CV_MODEL_PATH = CV_DIR / "cv_model.pt"
CV_METRICS_PATH = CV_DIR / "cv_metrics.json"


def find_busi_root(downloaded_path):
    """Find the folder containing benign/, malignant/, and normal/."""
    p = Path(downloaded_path)
    candidates = [p] + [x for x in p.rglob("*") if x.is_dir()]
    for candidate in candidates:
        if all((candidate / cls).is_dir() for cls in ["benign", "malignant", "normal"]):
            return candidate
    return None


def train_cv_from_kaggle():
    """Download BUSI through KaggleHub and train the bundled CNN."""
    import subprocess
    import sys
    import kagglehub

    with st.status("Preparing the ultrasound Computer Vision model...", expanded=True) as status:
        st.write("1/3 Downloading BUSI breast-ultrasound dataset from Kaggle...")
        downloaded = kagglehub.dataset_download(
            "subhajournal/busi-breast-ultrasound-images-dataset"
        )

        data_root = find_busi_root(downloaded)
        if data_root is None:
            status.update(
                label="BUSI dataset structure was not found.",
                state="error",
            )
            st.error(
                "I downloaded the dataset, but could not find benign/, malignant/, "
                "and normal/ folders."
            )
            return False

        st.write(f"Dataset found: `{data_root}`")
        st.write("2/3 Training the CNN. This may take several minutes on CPU...")

        cmd = [
            sys.executable,
            str(CV_DIR / "train_cv.py"),
            "--data", str(data_root),
            "--epochs", "8",
            "--batch-size", "32",
            "--image-size", "128",
            "--output", str(CV_DIR),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            st.code(result.stdout)

        if result.returncode != 0:
            if result.stderr:
                st.error(result.stderr)
            status.update(
                label="Computer Vision training failed.",
                state="error",
            )
            return False

        st.write("3/3 Model files created.")
        status.update(
            label="Computer Vision model installed successfully!",
            state="complete",
        )

    return CV_MODEL_PATH.exists()


if not CV_MODEL_PATH.exists():
    st.info(
        "🧠 The ultrasound Computer Vision model is not installed yet. "
        "Click the button below. Hiwet will download the BUSI dataset and "
        "train the CNN automatically."
    )

    if st.button(
        "🧠 Install / Train Ultrasound Computer Vision Model",
        type="primary",
        use_container_width=True,
    ):
        try:
            if train_cv_from_kaggle():
                st.success("✅ Ultrasound Computer Vision model is ready.")
                st.rerun()
        except Exception as exc:
            st.error(f"Computer Vision setup failed: {exc}")
else:
    try:
        from cv.predict_cv import predict_image

        st.success("✅ Ultrasound Computer Vision model is installed.")

        ultrasound = st.file_uploader(
            "Upload ultrasound image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="ultrasound_image",
        )

        if ultrasound is not None:
            st.image(
                ultrasound,
                caption="Uploaded ultrasound image",
                use_container_width=True,
            )

            if st.button(
                "🩻 Analyze ultrasound image",
                type="primary",
                use_container_width=True,
            ):
                try:
                    result = predict_image(ultrasound)
                    label = result["label"].upper()
                    confidence = max(result["probabilities"].values()) * 100

                    if label == "MALIGNANT":
                        st.error(
                            f"### ⚠️ CNN result: {label}\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )
                    elif label == "BENIGN":
                        st.success(
                            f"### ✅ CNN result: {label}\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )
                    else:
                        st.info(
                            f"### ℹ️ CNN result: {label}\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )

                    probs = pd.DataFrame({
                        "Class": list(result["probabilities"].keys()),
                        "Estimated probability (%)": [
                            v * 100 for v in result["probabilities"].values()
                        ],
                    })
                    st.bar_chart(
                        probs.set_index("Class"),
                        height=300,
                    )

                    st.warning(
                        "This is an AI image-classification estimate for "
                        "research/education. It is not a medical diagnosis "
                        "and must be reviewed by a qualified healthcare "
                        "professional."
                    )
                except Exception as exc:
                    st.error(f"Ultrasound analysis failed: {exc}")

    except Exception as exc:
        st.error(f"Could not load the Computer Vision module: {exc}")

st.divider()
st.caption("Hiwet Breast Cancer AI • SVM (clinical measurements) + CNN (ultrasound) • Educational/research use only")


# ============================================================
# HIWET: Confidence Gate for Ultrasound Predictions
# ============================================================
CONFIDENCE_THRESHOLD = 0.60

def display_prediction_with_confidence_gate(result):
    """
    Use this immediately after result = predict_image(uploaded_file).
    Predictions below the threshold are displayed as uncertain.
    """
    import pandas as pd

    predicted_class = result["label"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ Low Confidence / Uncertain Prediction — "
            "Medical Review Required"
        )
        st.write(
            f"AI confidence: **{confidence * 100:.2f}%** "
            f"(threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%)"
        )
    else:
        st.subheader(f"Prediction: {predicted_class}")
        st.metric("AI confidence", f"{confidence * 100:.2f}%")

    df = pd.DataFrame({
        "Class": list(probabilities.keys()),
        "Probability (%)": [v * 100 for v in probabilities.values()],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info(
        "⚠️ Research/decision-support only. This system does not "
        "provide a medical diagnosis. A qualified healthcare "
        "professional must review the ultrasound."
    )
