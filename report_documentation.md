# NLP-Based Domain-Specific Document Classifier
### Final Year Project Report and Documentation

**Course:** BCA (AI & Data Science)  **Student:** `<Your Name>`  **Roll No.:** `<Roll No.>`
**Guide:** `<Guide Name>`  **Institute:** `<College / University>`  **Year:** `<Year>`

> Fill the `<...>` fields before submission. Section 11 contains result numbers: replace them with the numbers
> printed on **your** machine (`models/accuracy_report.txt`), because small differences are normal.

---

## Table of Contents
1. Abstract
2. Introduction, Problem Statement, Objectives
3. Literature Survey
4. System Requirements
5. System Architecture Diagram
6. Data Flow Diagrams
7. Module Description
8. Methodology (NLP + ML theory)
9. Dataset Description
10. Implementation Summary
11. Results and Testing
12. Limitations
13. Future Scope
14. Conclusion
15. IEEE-Style Paper
16. Viva Questions and Answers
17. PPT Slide Content

---

## 1. Abstract

Organizations receive thousands of documents every day, and sorting them manually into categories is slow, costly
and error-prone. This project presents an NLP-based Domain-Specific Document Classifier that automatically identifies
the domain of an uploaded document: Medical, Finance, Legal, Technology, Education or Sports. The system accepts TXT,
PDF and DOCX files, extracts their text, and applies NLP preprocessing (lowercasing, tokenization, stopword removal and
lemmatization). The cleaned text is converted into numerical features with TF-IDF and classified by a machine-learning
model. Two models, Multinomial Naive Bayes and Logistic Regression, are trained, evaluated with Accuracy, Precision,
Recall, F1-score, Classification Report and Confusion Matrix, and compared. The better model is saved with Joblib and
used for prediction. For each document the system reports a confidence score, rejects unfamiliar content as "Unknown",
and stores the file automatically in a category folder. A Streamlit web application provides upload, text preview,
prediction, dashboard and model-performance views. The project demonstrates how classical NLP and machine learning can
automate document organization with a lightweight, explainable and easily deployable solution.

**Keywords:** Document classification, NLP, TF-IDF, Naive Bayes, Logistic Regression, Streamlit, Text mining.

---

## 2. Introduction

### 2.1 Background
Documents such as medical reports, financial statements, legal agreements, technical articles, academic notes and sports
news arrive in mixed batches. Text classification, a core NLP task, assigns each document to a predefined class
based on its content.

### 2.2 Problem Statement
Manually sorting documents into categories is time-consuming and inefficient. The system should analyze the content of a
document and classify it into the correct domain automatically.

### 2.3 Objectives
1. Extract text from TXT, PDF and DOCX documents.
2. Apply NLP preprocessing to clean the text.
3. Convert text to features using TF-IDF.
4. Train and compare Multinomial Naive Bayes and Logistic Regression.
5. Predict the domain with a confidence score and handle unknown documents.
6. Store each document automatically in the predicted category folder.
7. Provide a professional Streamlit interface and dashboard.

### 2.4 Scope
Six domains, three file formats, English text, text-based (not scanned) documents.

---

## 3. Literature Survey

| Work | Idea | Relevance |
|---|---|---|
| Salton and Buckley (1988) | TF-IDF term weighting | Feature extraction used in this project |
| McCallum and Nigam (1998) | Naive Bayes event models for text | Basis for Multinomial Naive Bayes |
| Joachims (1998) | Linear classifiers for text categorization | Supports use of linear models such as Logistic Regression |
| Sebastiani (2002) | Survey of ML for text categorization | Overall methodology |
| Manning et al. (2008) | Information retrieval textbook | Preprocessing and evaluation concepts |

Modern approaches use deep learning (BERT and other transformers). They are more accurate on very large data but need
more data, memory and training time. For a small, explainable, fast system, TF-IDF with linear models remains a strong
and practical baseline.

---

## 4. System Requirements

**Hardware:** any laptop, 4 GB RAM minimum, 500 MB free disk space.
**Software:** Python 3.9-3.12, Windows/macOS/Linux, a web browser.
**Libraries:** pandas, numpy, scikit-learn, nltk, streamlit, joblib, matplotlib, seaborn, PyPDF2, python-docx.

---

## 5. System Architecture Diagram

```
+--------------------------------------------------------------------------------+
|                          PRESENTATION LAYER  (app.py)                          |
|   Streamlit UI:  Upload | Text Preview | Predict Button | Result | Dashboard    |
+----------------------------------------+---------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------------------------+
|                    APPLICATION LAYER  (predict.py, utils.py)                   |
|     File Validation  ->  Text Extraction (TXT / PDF / DOCX)  ->  Workflow       |
+----------------------------------------+---------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------------------------+
|                         NLP LAYER  (preprocessing.py)                          |
|   Lowercase -> Clean -> Tokenize -> Stopword Removal -> Lemmatization          |
+----------------------------------------+---------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------------------------+
|                           ML LAYER  (domain_classifier.pkl)                    |
|   TF-IDF Vectorizer  ->  Best Classifier (Naive Bayes / Logistic Regression)   |
|   Output: domain + probability for each class                                  |
+----------------------------------------+---------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------------------------+
|                    DECISION AND STORAGE LAYER  (utils.py)                      |
|   Unknown check (length, vocabulary coverage, confidence threshold)            |
|   Create folder if missing -> Save file in categorized_documents/<Domain>/     |
+--------------------------------------------------------------------------------+

OFFLINE TRAINING PIPELINE (train_model.py)
  sample_dataset.csv -> Clean -> Preprocess -> Train/Test Split (80/20)
     -> TF-IDF + Naive Bayes ----\
                                   >-> Evaluate + Compare -> Save best model (Joblib)
     -> TF-IDF + Logistic Reg. --/
```

---

## 6. Data Flow Diagrams

### 6.1 Level 0 (Context Diagram)

```
                 document (TXT/PDF/DOCX)
   +--------+  ------------------------------>  +---------------------------+
   |  USER  |                                   |   DOMAIN CLASSIFICATION   |
   |        |  <------------------------------  |          SYSTEM           |
   +--------+  domain, confidence, stored path  +---------------------------+
```

### 6.2 Level 1

```
 [USER]
   | 1. upload file
   v
 (1.0 Validate file type) --invalid--> error message --> [USER]
   | valid
   v
 (2.0 Extract text) --empty / corrupt--> error message --> [USER]
   | raw text
   v
 (3.0 NLP preprocessing)
   | clean tokens
   v
 (4.0 TF-IDF + classifier) <------ read ------ [D1: domain_classifier.pkl]
   | domain + probabilities
   v
 (5.0 Unknown check)
   | final label (Domain or Unknown)
   v
 (6.0 Folder organizer) ---- save file ----> [D2: categorized_documents/<Domain>/]
   | storage path
   v
 (7.0 Display result) --> [USER]: category, confidence, success message, path

 TRAINING FLOW:  [D3: sample_dataset.csv] -> (T1 Clean) -> (T2 Preprocess) -> (T3 Train + Evaluate)
                 -> (T4 Compare, save best) -> [D1: domain_classifier.pkl] + metrics.json
```

---

## 7. Module Description

| Module | File | Responsibility |
|---|---|---|
| Preprocessing | `preprocessing.py` | Lowercase, clean, tokenize, remove stopwords, lemmatize |
| Utilities | `utils.py` | Settings, file readers, folder creation, storage, charts |
| Training | `train_model.py` | Dataset cleaning, TF-IDF, model training, evaluation, saving |
| Prediction | `predict.py` | Model loading, prediction, unknown handling, full workflow |
| Web App | `app.py` | Streamlit interface and dashboard |
| Dataset | `dataset/sample_dataset.csv` | 120 labelled documents |

---

## 8. Methodology

### 8.1 NLP Preprocessing
1. **Lowercasing:** "Court" and "court" become the same word.
2. **Cleaning:** URLs, e-mails, digits and punctuation are removed.
3. **Tokenization:** text is split into words.
4. **Stopword removal:** words such as "the", "is", "of" are removed because they carry no domain meaning.
5. **Lemmatization:** words are reduced to dictionary form ("patients" to "patient", "treating" to "treat").

### 8.2 TF-IDF Feature Extraction
TF-IDF gives a high weight to words that are frequent in one document but rare across all documents.

```
tf-idf(t, d) = (1 + ln tf(t, d)) x idf(t)          (sublinear TF)
idf(t)       = ln( (1 + n) / (1 + df(t)) ) + 1      (n = number of documents)
```
Each document vector is L2-normalized. The project uses unigrams and bigrams (single words and two-word phrases
such as "blood pressure").

### 8.3 Multinomial Naive Bayes
Applies Bayes' theorem with the assumption that words are independent given the class:
```
P(c | d)  is proportional to  P(c) x  product over words w of  P(w | c)^(weight of w in d)
P(w | c) = ( count(w, c) + alpha ) / ( total words in c + alpha x |V| )      (alpha = Laplace smoothing)
```
Fast, works well on small data, but its probabilities can be over-confident.

### 8.4 Logistic Regression
A linear model that learns a weight vector for every class and converts scores into probabilities (softmax):
```
P(c | x) = exp( w_c . x + b_c ) / sum over k of exp( w_k . x + b_k )
```
Learns which words push a document toward each class; usually gives better-calibrated confidence scores.

### 8.5 Evaluation Metrics
```
Accuracy  = correct predictions / all predictions
Precision = TP / (TP + FP)       "of the documents predicted as X, how many really are X"
Recall    = TP / (TP + FN)       "of the real X documents, how many were found"
F1-score  = 2 x Precision x Recall / (Precision + Recall)
```
The **confusion matrix** shows, for each actual domain (rows), how many documents were predicted as each domain (columns).
Weighted averages are used for multi-class precision, recall and F1. 5-fold cross-validation adds a more reliable
estimate on a small dataset.

### 8.6 Unknown-Document Handling
A document is labelled **Unknown** if (a) it has fewer than 5 meaningful words, (b) fewer than 10 % of its words are in
the training vocabulary, or (c) the highest class probability is below the threshold (default 0.40). Unknown documents
go to `categorized_documents/Unknown/` for manual review.

---

## 9. Dataset Description
* File: `dataset/sample_dataset.csv`, columns `text`, `label`.
* 120 documents, **20 per domain** (balanced classes), each 1-2 sentences of realistic domain text.
* Cleaning: missing values, empty text, duplicates and unknown labels are removed; labels are normalized.
* Split: 80 % training (96) and 20 % testing (24), stratified, `random_state = 42`.

---

## 10. Implementation Summary
1. `train_model.py` loads and cleans data, preprocesses text, trains both pipelines (TF-IDF + classifier), evaluates them,
   selects the best by test F1 (cross-validation F1 breaks ties), re-trains it on all data and saves it with Joblib.
2. `predict.py` loads the Joblib bundle once, applies the same preprocessing, computes class probabilities, applies the
   unknown rules and returns the result.
3. `utils.py` extracts text (PyPDF2 for PDF, python-docx for DOCX, decoding for TXT), creates folders and saves files
   without overwriting.
4. `app.py` builds the interface with tabs, metrics, progress bar and charts.

---

## 11. Results and Testing

### 11.1 Model Comparison (sample run)
Run `python train_model.py` and copy your own numbers from `models/accuracy_report.txt`.
In the author's test run (test set of 24 documents) the results were:

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | 5-fold CV F1 |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 95.8 % | 96.7 % | 95.8 % | 95.8 % | 88.7 % |
| Logistic Regression | 95.8 % | 96.7 % | 95.8 % | 95.8 % | 89.7 % |

Logistic Regression was selected because it tied on the test set and had the higher cross-validation F1.
*Note:* that run used the built-in stopword list without lemmatization (no internet in the test environment); on your
machine with full NLTK data, numbers may differ by a few points. The test set is small (4 documents per domain), so one
mistake changes accuracy by about 4 %. This is why cross-validation is reported as well. Confusion-matrix images are saved in `models/`.

### 11.2 Functional Test Cases

| No. | Input | Expected | Observed (author's run) |
|---|---|---|---|
| 1 | TXT about a patient and antibiotics | Medical | Medical (about 80 % confidence) |
| 2 | DOCX about a doctor and treatment | Medical | Medical |
| 3 | PDF about bank earnings and interest rates | Finance | Finance |
| 4 | Court judgment text | Legal | Legal |
| 5 | Cricket/football match report | Sports | Sports |
| 6 | Cake recipe (unseen domain) | Unknown | Unknown (low confidence) |
| 7 | Text: "Hello world" | Unknown (too short) | Unknown |
| 8 | Mixed hospital/court/company text | Unknown (ambiguous) | Unknown |
| 9 | Corrupt PDF / unsupported .exe / empty TXT | Friendly error | Friendly error |
| 10 | Same file uploaded twice | Both stored | Second file saved as `name_1.ext` |

---

## 12. Limitations
* Small dataset: accuracy on long, real-world documents will be lower until more data is added.
* English only; six fixed domains.
* Scanned/image-only PDFs are not supported (no OCR).
* TF-IDF ignores word order and deep context, so subtle documents can be confused.

## 13. Future Scope
1. Larger real-world datasets (BBC News, Kaggle) and more domains.
2. Deep learning models: BERT / DistilBERT embeddings for context-aware classification.
3. OCR (Tesseract) for scanned PDFs and images.
4. Multi-language support (Hindi and other languages).
5. Multi-label classification (a document that is both Legal and Finance).
6. Probability calibration and active learning (user corrects wrong predictions, system retrains).
7. Database storage (SQLite/PostgreSQL) with search, tags and metadata.
8. Batch upload and drag-and-drop folders; REST API (FastAPI) and cloud deployment (Docker).
9. User login, audit logs and role-based access for organizations.
10. Document summarization and keyword extraction for each stored file.

## 14. Conclusion
The project delivers a complete, working pipeline: file upload, text extraction, NLP preprocessing, TF-IDF features,
model comparison, confidence-based prediction, unknown-document handling and automatic folder organization, all inside
a simple Streamlit application. It shows that a classical NLP and machine-learning approach is accurate, fast, explainable
and practical for automating document sorting, and it provides a clear path for scaling with more data and modern models.

---

## 15. IEEE-Style Paper

**NLP-Based Domain-Specific Document Classification Using TF-IDF, Naive Bayes and Logistic Regression**

`<Author Name>`
Department of Computer Applications (AI & Data Science), `<Institute>`, `<City, Country>`
`<email@example.com>`

**Abstract**—Manual sorting of organizational documents is slow and error-prone. This paper presents an automated
system that classifies TXT, PDF and DOCX documents into six domains (Medical, Finance, Legal, Technology, Education and
Sports). Text is extracted and normalized using lowercasing, tokenization, stopword removal and lemmatization, then
represented with TF-IDF unigram and bigram features. Multinomial Naive Bayes and Logistic Regression classifiers are
trained and compared using accuracy, precision, recall, F1-score and confusion matrices. The best model is persisted with
Joblib and served through a Streamlit web application that reports confidence scores, rejects unfamiliar documents as
"Unknown" and automatically files each document in its category folder. Experiments on a balanced 120-document dataset
achieved about 96 % test accuracy for both models, with Logistic Regression preferred for its higher cross-validation F1.

**Index Terms**—Document classification, natural language processing, TF-IDF, Naive Bayes, logistic regression, Streamlit.

### I. Introduction
Organizations process large volumes of heterogeneous documents, and routing them to the right department or archive
is a repetitive task. Automatic text classification [7] addresses this by assigning each document to a predefined
category based on content. This work builds an end-to-end, lightweight classifier that can be run on an ordinary computer,
and couples it with automatic document storage and a user-friendly interface.

### II. Related Work
Term weighting by TF-IDF [3] remains a standard text representation. Naive Bayes classifiers for text were analyzed by
McCallum and Nigam [4], and linear classifiers were shown to perform strongly on high-dimensional sparse text by
Joachims [5]. Sebastiani [7] surveys machine-learning approaches to text categorization, and Manning et al. [6]
describe preprocessing and evaluation practice. Transformer models achieve higher accuracy on large data, but at greater
computational cost; this work targets a fast, interpretable baseline.

### III. Proposed System
The system has five stages: (1) document upload and text extraction using PyPDF2 and python-docx; (2) NLP preprocessing
with NLTK [2]; (3) TF-IDF feature extraction; (4) classification with the selected model; (5) confidence evaluation and
automatic storage. Unknown documents are detected using three rules: minimum length, vocabulary coverage and a probability
threshold.

### IV. Methodology
**A. Preprocessing.** Text is lowercased, URLs and non-alphabetic characters are removed, tokens are extracted,
English stopwords are removed and tokens are lemmatized with WordNet.
**B. Features.** Sublinear TF-IDF with unigrams and bigrams: tf-idf(t,d) = (1 + ln tf) x idf(t), with idf(t) = ln((1+n)/(1+df(t))) + 1.
**C. Classifiers.** Multinomial Naive Bayes (alpha = 0.1) and multinomial Logistic Regression (C = 10, L2 regularization),
both implemented in scikit-learn [1].
**D. Evaluation.** Stratified 80/20 split, plus 5-fold stratified cross-validation. Metrics: accuracy, weighted precision,
recall and F1, classification report and confusion matrix.

### V. Experimental Setup
The dataset contains 120 documents, 20 per class. After cleaning, 96 documents were used for training and 24 for testing.
The random seed was fixed (42) for reproducibility. Implementation language: Python 3.

### VI. Results and Discussion
TABLE I. PERFORMANCE COMPARISON (TEST SET, 24 DOCUMENTS)

| Model | Accuracy | Precision | Recall | F1 | CV F1 |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.958 | 0.967 | 0.958 | 0.958 | 0.887 |
| Logistic Regression | 0.958 | 0.967 | 0.958 | 0.958 | 0.897 |

Both models performed equally on the small test set; the confusion matrices showed that the few errors occurred between
domains with overlapping vocabulary. In the author's run, an accounting document (ledger, transaction, trial balance) was
predicted as Technology because words such as "ledger" and "transaction" also occur in the blockchain training sample.
Logistic Regression was
selected due to its higher cross-validation F1 and better-calibrated probabilities, which makes the confidence-based
rejection of unknown documents more reliable. Test-set size limits statistical certainty, so larger datasets are needed to
confirm the difference between the models. `<Replace the numbers above with the values from your run.>`

### VII. Conclusion and Future Work
An end-to-end document classifier combining NLP preprocessing, TF-IDF and linear models was implemented with automatic
folder organization and a web dashboard. Future work includes larger datasets, transformer-based embeddings, OCR for
scanned documents, multi-label classification and multilingual support.

### References
[1] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.
[2] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. Sebastopol, CA, USA: O'Reilly Media, 2009.
[3] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513-523, 1988.
[4] A. McCallum and K. Nigam, "A comparison of event models for Naive Bayes text classification," in *Proc. AAAI-98 Workshop on Learning for Text Categorization*, 1998, pp. 41-48.
[5] T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," in *Proc. ECML*, 1998, pp. 137-142.
[6] C. D. Manning, P. Raghavan, and H. Schutze, *Introduction to Information Retrieval*. Cambridge, U.K.: Cambridge Univ. Press, 2008.
[7] F. Sebastiani, "Machine learning in automated text categorization," *ACM Comput. Surveys*, vol. 34, no. 1, pp. 1-47, 2002.

---

## 16. Viva Questions and Answers

**Q1. What is your project about?**
It automatically identifies the domain of an uploaded document (Medical, Finance, Legal, Technology, Education or Sports) using NLP and machine learning, shows a confidence score and stores the file in the right folder.

**Q2. What is NLP?**
Natural Language Processing is the field of AI that lets computers understand and process human language.

**Q3. What is text classification?**
Assigning a piece of text to one of several predefined categories based on its content.

**Q4. Why do we preprocess text?**
Raw text contains noise (case differences, punctuation, filler words). Cleaning it reduces noise and vocabulary size so the model learns real patterns.

**Q5. Explain each preprocessing step.**
Lowercasing unifies words; tokenization splits text into words; stopword removal deletes common words like "the"; lemmatization converts words to their base form ("studies" to "study").

**Q6. Stemming vs lemmatization?**
Stemming chops word endings by rules and may produce non-words ("studi"); lemmatization uses a dictionary (WordNet) and returns real words. I used lemmatization.

**Q7. What is TF-IDF?**
A weighting scheme: Term Frequency counts how often a word appears in a document, Inverse Document Frequency lowers the weight of words common to all documents. So domain-specific words like "court" get high scores and common words get low scores.

**Q8. Why TF-IDF instead of Bag of Words?**
Bag of Words only counts words, so frequent but unimportant words dominate. TF-IDF highlights distinctive words and usually classifies better.

**Q9. What are n-grams? Why did you use (1,2)?**
N-grams are sequences of n words. Bigrams capture phrases such as "blood pressure" or "interest rate" that carry more meaning than the separate words.

**Q10. How does Multinomial Naive Bayes work?**
It uses Bayes' theorem and assumes words are independent given the class. It picks the class with the highest probability from the class prior and the word likelihoods.

**Q11. Why is it called "naive"?**
Because it assumes word independence, which is not true in real language, but it still works well in practice.

**Q12. How does Logistic Regression work for multiple classes?**
It learns a weight for every word per class, computes a score per class and converts scores to probabilities with the softmax function.

**Q13. Which model is better and why?**
In my run both had the same test accuracy; Logistic Regression had slightly higher cross-validation F1 and better-calibrated probabilities, so it was selected. The system chooses automatically from the results.

**Q14. Explain accuracy, precision, recall and F1.**
Accuracy = correct / total. Precision = of the documents predicted as a class, how many were correct. Recall = of the actual documents of a class, how many were found. F1 = harmonic mean of precision and recall.

**Q15. What is a confusion matrix?**
A table where rows are actual classes and columns are predicted classes; the diagonal shows correct predictions, off-diagonal cells show which classes are confused.

**Q16. Why did you use train-test split and stratify?**
To test on unseen documents. Stratify keeps the same class proportions in both parts so each domain is tested fairly.

**Q17. What is cross-validation and why did you use it?**
The data is split into 5 parts; the model is trained 5 times, each time testing on a different part. With only 24 test documents, one mistake changes accuracy a lot, so cross-validation gives a more stable estimate.

**Q18. What is overfitting? How do you reduce it?**
When a model memorizes training data and fails on new data. Reduced by a held-out test set, cross-validation, regularization (Logistic Regression's C), and more data.

**Q19. What is Joblib and why use it?**
A library to save and load Python objects efficiently, especially scikit-learn models with large NumPy arrays. We save the trained pipeline once and load it for prediction without retraining.

**Q20. What is a Pipeline?**
A scikit-learn object that chains TF-IDF and the classifier so they are trained and applied together, which avoids mistakes and data leakage.

**Q21. How is the confidence score calculated?**
It is the highest probability from `predict_proba` for the predicted class (0 to 100 %).

**Q22. How do you handle unknown documents?**
Three checks: too little text, too few words known to the model, or confidence below the threshold (40 % default, adjustable). Such files go to the `Unknown` folder.

**Q23. How do you extract text from PDF and DOCX?**
PyPDF2 reads each PDF page's text; python-docx reads paragraphs and table cells; TXT files are decoded as UTF-8 with fallbacks.

**Q24. Can it read scanned PDFs?**
No. Scanned PDFs are images and need OCR (for example Tesseract). This is listed as future scope.

**Q25. How are folders created and files stored?**
`ensure_directories()` creates missing folders with `mkdir(parents=True, exist_ok=True)`. `store_document()` saves the file in `categorized_documents/<Domain>/` and adds a number to the name if the file already exists.

**Q26. What are the limitations?**
Small dataset, English only, six fixed domains, no OCR, and TF-IDF ignores word order.

**Q27. How can you improve accuracy?**
Add more and longer real documents, tune hyperparameters, try SVM or BERT embeddings, and add active learning from user corrections.

**Q28. Why Streamlit?**
It turns Python scripts into interactive web apps quickly without HTML/JavaScript, ideal for ML demos.

**Q29. Why is the same preprocessing used in training and prediction?**
If prediction text were cleaned differently, its features would not match what the model learned, and accuracy would fall.

**Q30. What is the difference between this project and deep learning approaches?**
Deep models like BERT understand context and can be more accurate but need more data, memory and training time. This project is lightweight, fast and explainable, which suits a small dataset and limited hardware.

---

## 17. PPT Slide Content

**Slide 1 - Title**
NLP-Based Domain-Specific Document Classifier. Student name, roll number, guide, college, year.

**Slide 2 - Problem Statement**
Organizations handle thousands of documents daily. Manual sorting is slow, inefficient and error-prone. Need: automatic domain identification.

**Slide 3 - Objectives**
Extract text from TXT/PDF/DOCX; clean text with NLP; classify into 6 domains; show confidence; auto-store in folders; web dashboard.

**Slide 4 - Proposed Solution and Workflow**
Upload -> Extract text -> NLP preprocessing -> TF-IDF -> ML prediction -> Confidence -> Auto-save in folder -> Show result.

**Slide 5 - System Architecture**
Use the diagram from Section 5 (Presentation, Application, NLP, ML, Storage layers).

**Slide 6 - Technology Stack**
Python, scikit-learn, NLTK, Streamlit, Joblib, PyPDF2, python-docx, pandas, matplotlib, seaborn.

**Slide 7 - NLP Preprocessing**
Lowercasing, tokenization, stopword removal, lemmatization, with one before/after example sentence.

**Slide 8 - Feature Engineering: TF-IDF**
Formula, intuition (distinctive words get higher weights), unigrams plus bigrams.

**Slide 9 - Machine Learning Models**
Multinomial Naive Bayes (probabilistic, fast) vs Logistic Regression (linear, better probabilities).

**Slide 10 - Dataset**
120 documents, 20 per domain, 80/20 stratified split, cleaning steps.

**Slide 11 - Results**
Metrics table (Accuracy, Precision, Recall, F1), comparison bar chart (`models/model_comparison.png`), best model highlighted.

**Slide 12 - Confusion Matrix**
Insert `models/confusion_matrix_logistic_regression.png`; explain diagonal vs off-diagonal.

**Slide 13 - Unknown Document Handling and Auto-Organization**
Three rules; `Unknown` folder; screenshot of `categorized_documents/` tree.

**Slide 14 - Live Demo (Streamlit)**
Screenshots: upload, preview, prediction with confidence, storage path, dashboard.

**Slide 15 - Limitations and Future Scope**
Small data, English only, no OCR. Future: BERT, OCR, multilingual, multi-label, database, API/cloud.

**Slide 16 - Conclusion and Thank You**
Working end-to-end system, accurate and explainable, ready to scale. Questions?
