"""
app.py
======
Streamlit web application for the NLP Domain-Specific Document Classifier.

Run:   streamlit run app.py

Tabs
----
 1. Classify Document  - upload, preview text, predict, confidence, storage path
 2. Dashboard          - documents per category, recent documents
 3. Model Performance  - Naive Bayes vs Logistic Regression, confusion matrices
 4. About              - project description
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import predict
import utils

# --------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Domain-Specific Document Classifier",
    page_icon="📄",
    layout="wide",
)
utils.ensure_directories()  # creates all category folders on startup

DOMAIN_ICONS = {
    "Medical": "🩺",
    "Finance": "💰",
    "Legal": "⚖️",
    "Technology": "💻",
    "Education": "🎓",
    "Sports": "🏅",
    "Unknown": "❓",
}

# Small bit of CSS for the header banner
st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #1f3c88 0%, #3a7bd5 60%, #00b4db 100%);
        color: white;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: 0.3rem 0 0 0; opacity: 0.92; }
    </style>
    <div class="hero">
        <h1>📄 NLP-Based Domain-Specific Document Classifier</h1>
        <p>Upload a TXT, PDF or DOCX file and the system identifies its domain, shows a
        confidence score and files the document automatically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# CACHED HELPERS (so Streamlit does not repeat heavy work on every click)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load the Joblib model once and keep it in memory."""
    return predict.load_model(force_reload=True)


@st.cache_data(show_spinner=False)
def cached_extract(data, name):
    """Extract text once per unique uploaded file."""
    return utils.extract_text(data, name)


def train_now():
    """Run the full training script from inside the app, then reload the model."""
    import train_model  # imported here so the app starts fast

    with st.spinner("Training models... this takes a few seconds."):
        train_model.main()
    get_model_bundle.clear()
    predict.load_model(force_reload=True)


# --------------------------------------------------------------------------
# MODEL MUST EXIST BEFORE THE APP CAN WORK
# --------------------------------------------------------------------------
if not predict.model_exists():
    st.warning("⚠️ No trained model was found (models/domain_classifier.pkl).")
    st.write("Click the button to train it now, or run `python train_model.py` in the terminal.")
    if st.button("🚀 Train model now", type="primary"):
        train_now()
        st.rerun()
    st.stop()

bundle = get_model_bundle()
metrics = utils.load_metrics()

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(utils.CONFIDENCE_THRESHOLD),
        step=0.05,
        help="Documents whose top probability is below this value are marked 'Unknown'.",
    )
    st.markdown("---")
    st.subheader("🤖 Active Model")
    st.write(f"**{bundle['model_name']}**")
    st.caption(f"Trained on {bundle['n_training_samples']} documents")
    st.markdown("**Domains**")
    for domain in utils.DOMAINS:
        st.write(f"{DOMAIN_ICONS[domain]} {domain}")
    st.markdown("---")
    if st.button("🔄 Retrain model"):
        train_now()
        st.rerun()

tab_classify, tab_dashboard, tab_performance, tab_about = st.tabs(
    ["📤 Classify Document", "📊 Dashboard", "🧪 Model Performance", "ℹ️ About"]
)


# --------------------------------------------------------------------------
# RESULT DISPLAY
# --------------------------------------------------------------------------
def show_result(result, stored_path):
    """Render the classification result: message, category, confidence, path."""
    domain = result["domain"]
    icon = DOMAIN_ICONS.get(domain, "📁")

    if result["is_unknown"]:
        st.warning("⚠️ This document could not be classified reliably. It was stored in the 'Unknown' folder.")
        st.caption(f"Reason: {result['reason']}")
    else:
        st.success(f"✅ Document classified successfully as **{domain}** and stored in its category folder.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Category", f"{icon} {domain}")
    col2.metric("Confidence Score", f"{result['confidence']:.1%}")
    if result["is_unknown"] and result["top_guess"] != "-":
        col3.metric("Closest Guess", result["top_guess"])
    else:
        col3.metric("Status", "Unknown" if result["is_unknown"] else "Classified")

    st.progress(float(min(max(result["confidence"], 0.0), 1.0)))

    st.markdown("**📁 Document Storage Path**")
    st.code(str(stored_path), language=None)

    if result["probabilities"]:
        st.markdown("**Probability for each domain**")
        prob_df = pd.DataFrame({"Probability": result["probabilities"]})
        st.bar_chart(prob_df)


# --------------------------------------------------------------------------
# TAB 1: CLASSIFY
# --------------------------------------------------------------------------
with tab_classify:
    uploaded = st.file_uploader("Upload a document (TXT, PDF or DOCX)", type=["txt", "pdf", "docx"])

    if uploaded is None:
        st.info("👆 Upload a document to begin.")
    else:
        file_bytes = uploaded.getvalue()
        text = None
        try:
            text = cached_extract(file_bytes, uploaded.name)
        except ValueError as error:  # unreadable / empty / corrupt file
            st.error(f"❌ {error}")

        if text:
            c1, c2, c3 = st.columns(3)
            c1.metric("File Type", Path(uploaded.name).suffix.upper().lstrip("."))
            c2.metric("Words", f"{len(text.split()):,}")
            c3.metric("Characters", f"{len(text):,}")

            st.subheader("📝 Extracted Text Preview")
            preview = text[:3000] + ("..." if len(text) > 3000 else "")
            st.text_area("Preview", value=preview, height=220, disabled=True, label_visibility="collapsed")

            if st.button("🔍 Predict Domain", type="primary"):
                with st.spinner("Analysing document..."):
                    result = predict.predict_domain(text, threshold)
                    stored = utils.store_document(file_bytes, uploaded.name, result["domain"])
                # Remember the result so it stays visible if the page reruns
                st.session_state["last_result"] = {"file": uploaded.name, "result": result, "path": stored}

            saved = st.session_state.get("last_result")
            if saved and saved["file"] == uploaded.name:
                st.markdown("---")
                st.subheader("🎯 Classification Result")
                show_result(saved["result"], saved["path"])


# --------------------------------------------------------------------------
# TAB 2: DASHBOARD
# --------------------------------------------------------------------------
with tab_dashboard:
    counts = utils.count_documents()
    best = metrics["best_model"] if metrics else bundle["model_name"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents Stored", sum(counts.values()))
    m2.metric("Domains Supported", len(utils.DOMAINS))
    m3.metric("Best Model", best)
    if metrics:
        m4.metric("Test Accuracy", f"{metrics['models'][best]['accuracy']:.1%}")
    else:
        m4.metric("Test Accuracy", "n/a")

    st.subheader("Documents per Category")
    card_cols = st.columns(len(counts))
    for col, (name, number) in zip(card_cols, counts.items()):
        col.metric(f"{DOMAIN_ICONS.get(name, '')} {name}", number)
    st.bar_chart(pd.DataFrame({"Documents": counts}))

    st.subheader("Recently Stored Documents")
    recent = utils.list_stored_documents(limit=15)
    if recent:
        st.dataframe(pd.DataFrame(recent), hide_index=True)
    else:
        st.info("No documents stored yet. Classify a document in the first tab.")


# --------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# --------------------------------------------------------------------------
with tab_performance:
    if not metrics:
        st.info("No metrics found. Click 'Retrain model' in the sidebar to generate them.")
    else:
        st.success(f"🏆 Best model: **{metrics['best_model']}** (saved with Joblib and used for predictions)")
        st.caption(
            f"Dataset: {metrics['n_samples']} documents | "
            f"Training: {metrics['n_train']} | Testing: {metrics['n_test']}"
        )

        score_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "CV F1 (mean)"]
        table = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy": r["accuracy"],
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "F1-Score": r["f1"],
                    "CV F1 (mean)": r["cv_f1_mean"],
                }
                for name, r in metrics["models"].items()
            ]
        )
        shown = table.copy()
        for column in score_cols:
            shown[column] = (table[column] * 100).round(2).astype(str) + "%"
        st.subheader("Model Comparison")
        st.dataframe(shown, hide_index=True)
        st.bar_chart(table.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].T)

        st.subheader("Confusion Matrices")
        matrix_cols = st.columns(len(metrics["models"]))
        for col, (name, r) in zip(matrix_cols, metrics["models"].items()):
            with col:
                figure = utils.plot_confusion_matrix(
                    np.array(r["confusion_matrix"]), r["labels"], name
                )
                st.pyplot(figure)
                plt.close(figure)

        st.subheader("Classification Reports")
        for name, r in metrics["models"].items():
            with st.expander(name):
                st.code(r["report"], language=None)


# --------------------------------------------------------------------------
# TAB 4: ABOUT
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this Project")
    st.write(
        "Organisations receive thousands of documents every day and sorting them by hand is slow. "
        "This system reads a document, cleans the text with NLP, predicts its domain with a "
        "machine-learning model and files it in the right folder automatically."
    )
    st.markdown(
        """
**Workflow**
1. Upload a TXT, PDF or DOCX document
2. Text is extracted from the file
3. NLP preprocessing: lowercasing, tokenization, stopword removal, lemmatization
4. TF-IDF converts the text into numbers
5. The best model (Naive Bayes or Logistic Regression) predicts the domain
6. Confidence score is shown; low-confidence documents become **Unknown**
7. The file is saved in `categorized_documents/<Domain>/`

**Technology:** Python, scikit-learn, NLTK, Streamlit, Joblib, PyPDF2, python-docx, pandas, matplotlib, seaborn

**Limitations:** scanned (image-only) PDFs need OCR and are not supported; the model only knows the six trained domains.
"""
    )
