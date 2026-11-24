"""
utils.py
========
Shared helpers for the whole project:

    * Project paths and settings (one place to change them)
    * Text extraction from TXT / PDF / DOCX files
    * Category-folder creation and document storage
    * Dashboard helpers (document counts, recent files)
    * Confusion-matrix plotting

Keeping these here keeps app.py, train_model.py and predict.py short and clean.
"""

import io
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-GUI backend: safe for servers and Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from PyPDF2 import PdfReader

# --------------------------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent            # folder containing this file
DATASET_PATH = BASE_DIR / "dataset" / "sample_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "domain_classifier.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
CATEGORY_ROOT = BASE_DIR / "categorized_documents"

DOMAINS = ["Medical", "Finance", "Legal", "Technology", "Education", "Sports"]
UNKNOWN_LABEL = "Unknown"                              # folder for unrecognised documents
SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")

# --- Unknown-document handling ---------------------------------------------
# A document is "Unknown" if ANY of these is true:
#   * fewer than MIN_TOKENS meaningful words remain after preprocessing
#   * fewer than MIN_VOCAB_COVERAGE of its words were seen during training
#   * the top class probability is below CONFIDENCE_THRESHOLD
MIN_TOKENS = 5
MIN_VOCAB_COVERAGE = 0.10
CONFIDENCE_THRESHOLD = 0.40


# --------------------------------------------------------------------------
# 2. FOLDER ORGANISATION
# --------------------------------------------------------------------------
def ensure_directories():
    """Create every folder the project needs (safe to call many times)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name in DOMAINS + [UNKNOWN_LABEL]:
        (CATEGORY_ROOT / name).mkdir(parents=True, exist_ok=True)


def safe_filename(filename):
    """Strip any path and replace risky characters so the file is saved safely."""
    name = Path(filename).name                        # removes folders like ../../
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)      # keep only safe characters
    return name or "document"


def store_document(data, filename, domain):
    """
    Save the uploaded file inside categorized_documents/<domain>/.
    If a file with the same name exists, a counter is added (report_1.pdf, ...)
    so nothing is ever overwritten. Returns the final Path.
    """
    ensure_directories()
    folder = CATEGORY_ROOT / (domain if domain in DOMAINS else UNKNOWN_LABEL)
    folder.mkdir(parents=True, exist_ok=True)

    name = safe_filename(filename)
    target = folder / name
    stem, suffix = target.stem, target.suffix
    counter = 1
    while target.exists():
        target = folder / f"{stem}_{counter}{suffix}"
        counter += 1

    target.write_bytes(data)
    return target


def count_documents():
    """Return {category: number_of_files} for the dashboard."""
    ensure_directories()
    counts = {}
    for name in DOMAINS + [UNKNOWN_LABEL]:
        folder = CATEGORY_ROOT / name
        counts[name] = sum(
            1 for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
        )  # hidden files such as .DS_Store are ignored
    return counts


def list_stored_documents(limit=15):
    """Return the most recently stored documents (newest first) as a list of dicts."""
    ensure_directories()
    rows = []
    for name in DOMAINS + [UNKNOWN_LABEL]:
        for path in (CATEGORY_ROOT / name).iterdir():
            if path.is_file() and not path.name.startswith("."):
                info = path.stat()
                rows.append(
                    {
                        "File": path.name,
                        "Category": name,
                        "Size (KB)": round(info.st_size / 1024, 1),
                        "Stored On": datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M"),
                        "_mtime": info.st_mtime,
                    }
                )
    rows.sort(key=lambda r: r["_mtime"], reverse=True)
    for r in rows:
        r.pop("_mtime")
    return rows[:limit]


# --------------------------------------------------------------------------
# 3. TEXT EXTRACTION (TXT / PDF / DOCX)
# --------------------------------------------------------------------------
def extract_text_from_txt(data):
    """Decode bytes of a text file, trying several common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_pdf(data):
    """Extract text from every page of a PDF using PyPDF2."""
    reader = PdfReader(io.BytesIO(data))

    if reader.is_encrypted:  # try an empty password, otherwise give up politely
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise ValueError("The PDF is password protected and cannot be read.")

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")  # extract_text() may return None
        except Exception:
            pages.append("")                          # skip unreadable page, keep going
    return "\n".join(pages)


def extract_text_from_docx(data):
    """Extract paragraphs AND table cells from a Word document."""
    document = Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def extract_text(data, filename):
    """
    Main entry point: choose the right extractor from the file extension.
    Raises ValueError with a friendly message for unsupported/empty/corrupt files.
    """
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{extension}'. Please upload TXT, PDF or DOCX.")

    try:
        if extension == ".txt":
            text = extract_text_from_txt(data)
        elif extension == ".pdf":
            text = extract_text_from_pdf(data)
        else:
            text = extract_text_from_docx(data)
    except ValueError:
        raise
    except Exception as exc:  # corrupt file, wrong format, etc.
        raise ValueError(f"Could not read '{filename}': {exc}") from exc

    text = text.strip()
    if not text:
        raise ValueError(
            "No readable text was found. Scanned/image-only PDFs need OCR, which is not supported."
        )
    return text


# --------------------------------------------------------------------------
# 4. METRICS + VISUALISATION
# --------------------------------------------------------------------------
def load_metrics():
    """Load the evaluation results saved by train_model.py (or None if missing)."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_confusion_matrix(cm, labels, title, save_path=None):
    """
    Draw a confusion matrix as a seaborn heatmap.
    Rows = true class, columns = predicted class. Returns the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Domain")
    ax.set_ylabel("Actual Domain")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
