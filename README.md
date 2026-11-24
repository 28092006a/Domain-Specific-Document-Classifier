# 📄 NLP-Based Domain-Specific Document Classifier

An intelligent system that reads an uploaded **TXT / PDF / DOCX** document, cleans the text with NLP,
predicts its **domain** (Medical, Finance, Legal, Technology, Education, Sports) using Machine Learning,
shows a **confidence score**, and **automatically files the document** in the matching folder.

---
## 1. Project Structure

```
Domain_Classifier/
├── dataset/
│   └── sample_dataset.csv        120 labelled documents (20 per domain)
├── categorized_documents/        Uploaded files are stored here, one folder per domain
│   ├── Medical/ Finance/ Legal/ Technology/ Education/ Sports/
│   └── Unknown/                  (extra) low-confidence / unrecognised documents
├── models/
│   └── domain_classifier.pkl     Created by train_model.py (also metrics.json + PNG charts)
├── app.py                        Streamlit web application
├── train_model.py                Dataset cleaning, training, evaluation, model saving
├── predict.py                    Loads the model and classifies documents
├── preprocessing.py              NLP text-cleaning pipeline
├── utils.py                      Paths, file readers, folder organiser, plots
├── requirements.txt
├── README.md
└── report_documentation.md       Report, abstract, diagrams, IEEE paper, viva Q&A, PPT content
```

---

## 2. Installation

You need **Python 3.9 – 3.12** and an internet connection for the first run
(NLTK downloads two small data packages automatically).

**Step 1 – Open a terminal inside the `Domain_Classifier` folder.**

**Step 2 – Create a virtual environment (recommended).**

| System | Commands |
|---|---|
| Windows | `python -m venv venv` then `venv\Scripts\activate` |
| macOS / Linux | `python3 -m venv venv` then `source venv/bin/activate` |

**Step 3 – Install the libraries.**

```bash
pip install -r requirements.txt
```

**Step 4 (optional) – Download the NLTK data now** instead of on first run:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 3. Execution Steps

**Step 1 – Train the model** (about 10 seconds):

```bash
python train_model.py
```

This prints the accuracy, precision, recall, F1, classification report and confusion matrix of both
models, picks the best one, and saves everything inside `models/`.

**Step 2 – Start the web app:**

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

**Step 3 – Classify a document:** upload a file → read the text preview → click **Predict Domain** →
see the category, confidence, success message and the folder path where the file was saved.

**Optional – command-line prediction:**

```bash
python predict.py path/to/document.pdf
```

> If you open the app before training, it shows a **Train model now** button, so you can also train from the browser.

---

## 4. File-by-File Explanation

### `preprocessing.py` – text cleaning
`preprocess_text(text)` turns raw text into clean text in five steps: lowercase → remove URLs/e-mails →
tokenize (letters only) → remove stopwords → lemmatize (`studies → study`).
It is used by **both** training and prediction so the model always sees data cleaned the same way.
If NLTK data cannot be downloaded, it falls back to a built-in stopword list instead of crashing.

### `utils.py` – shared helpers
| Function | Purpose |
|---|---|
| `ensure_directories()` | Creates `models/` and every category folder if missing |
| `extract_text(data, filename)` | Chooses the TXT / PDF / DOCX reader; raises friendly errors for corrupt, empty, password-protected or unsupported files |
| `store_document(data, filename, domain)` | Saves the file in `categorized_documents/<domain>/`, never overwriting (adds `_1`, `_2`…) |
| `count_documents()`, `list_stored_documents()` | Feed the dashboard |
| `plot_confusion_matrix()` | Seaborn heatmap used in training and in the app |
| Settings block | Paths, domain list, and the unknown-document thresholds |

### `train_model.py` – machine learning
1. Loads `sample_dataset.csv`, removes empty rows/duplicates/unknown labels, preprocesses the text.
2. Splits 80 % train / 20 % test (stratified, `random_state=42`).
3. Builds two pipelines: **TF-IDF + Multinomial Naive Bayes** and **TF-IDF + Logistic Regression**.
4. Evaluates both: Accuracy, Precision, Recall, F1, Classification Report, Confusion Matrix, plus 5-fold cross-validation.
5. Selects the best model, re-trains it on all data, and saves it with **Joblib** to `models/domain_classifier.pkl`.
6. Writes `metrics.json`, `accuracy_report.txt`, confusion-matrix PNGs and `model_comparison.png`.

### `predict.py` – prediction
`load_model()` loads the `.pkl`; `predict_domain(text)` returns the domain, confidence and all class
probabilities; `classify_file(bytes, name)` runs the whole workflow (extract → predict → store).

**Unknown-document handling** – a document becomes `Unknown` (stored in `categorized_documents/Unknown/`) when:
* it has fewer than 5 meaningful words, or
* fewer than 10 % of its words appeared in the training vocabulary, or
* the top probability is below the confidence threshold (default 40 %, adjustable in the app sidebar).

### `app.py` – Streamlit interface
Banner, upload button, extracted-text preview, **Predict Domain** button, success message, predicted
category, confidence score + progress bar, probability chart, storage path, and three tabs:
**Dashboard** (counts per folder, recent documents), **Model Performance** (model comparison, confusion matrices,
reports) and **About**.

---

## 5. Improving Accuracy (important for your viva)

The included dataset is small (120 documents) so it works well for a demo, but real documents are longer and messier.
To make the project stronger:

1. **Add more data.** Append rows to `dataset/sample_dataset.csv` (`text,label`) – ideally 200+ per domain – then run `python train_model.py` again. Public sources: BBC News, 20 Newsgroups, Kaggle article datasets.
2. **Use longer documents** in the dataset (full paragraphs/pages), because uploaded files are usually long.
3. **Tune the threshold** with the sidebar slider if too many good documents become "Unknown".

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| NLTK download fails (no internet) | The app still works (built-in stopwords, no lemmatization). Connect to the internet once and retrain for full NLP. |
| "No readable text was found" for a PDF | The PDF is a scanned image. OCR (e.g. Tesseract) would be needed – listed under Future Scope. |
| Model not found | Run `python train_model.py` (or use the **Train model now** button). |
| Warning about scikit-learn version when loading the model | Delete `models/domain_classifier.pkl` and run `python train_model.py` again on your machine. |
| `streamlit` command not found | Use `python -m streamlit run app.py` |
| Changed `preprocessing.py` | Always retrain afterwards so training and prediction use the same cleaning. |
