"""
predict.py
==========
Loads the trained model and classifies documents.

Main functions
--------------
load_model()              -> loads domain_classifier.pkl (cached after first load)
predict_domain(text)      -> domain + confidence + all class probabilities
classify_file(data, name) -> extract text, predict, and store the file in its folder

Command line:   python predict.py path/to/document.pdf
"""

import sys
from pathlib import Path

import joblib

import utils
from preprocessing import preprocess_tokens

_bundle = None  # in-memory cache so the .pkl is read from disk only once


class ModelNotTrainedError(FileNotFoundError):
    """Raised when models/domain_classifier.pkl does not exist yet."""


def model_exists():
    return utils.MODEL_PATH.exists()


def load_model(force_reload=False):
    """Load the saved bundle {pipeline, model_name, classes, ...} with Joblib."""
    global _bundle
    if _bundle is None or force_reload:
        if not model_exists():
            raise ModelNotTrainedError(
                "Model file not found. Run 'python train_model.py' first."
            )
        _bundle = joblib.load(utils.MODEL_PATH)
    return _bundle


def _unknown_result(reason, top_label="-", confidence=0.0, probabilities=None):
    """Build the standard result dictionary for an 'Unknown' document."""
    return {
        "domain": utils.UNKNOWN_LABEL,
        "top_guess": top_label,
        "confidence": float(confidence),
        "probabilities": probabilities or {},
        "is_unknown": True,
        "reason": reason,
    }


def predict_domain(text, threshold=None):
    """
    Predict the domain of raw document text.

    Returns a dictionary:
        domain        - predicted domain, or 'Unknown'
        top_guess     - the model's best guess even when the result is Unknown
        confidence    - probability of the top class (0.0 - 1.0)
        probabilities - {domain: probability} for all domains, sorted high -> low
        is_unknown    - True if the document could not be classified reliably
        reason        - short explanation (shown to the user when Unknown)
    """
    threshold = utils.CONFIDENCE_THRESHOLD if threshold is None else threshold
    pipeline = load_model()["pipeline"]

    # Same preprocessing as training
    tokens = preprocess_tokens(text)
    if len(tokens) < utils.MIN_TOKENS:
        return _unknown_result("The document contains too little meaningful text.")

    # Vocabulary check: how many words did the model see during training?
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    known_words = sum(1 for t in tokens if t in vocabulary)
    coverage = known_words / len(tokens)
    if known_words == 0 or coverage < utils.MIN_VOCAB_COVERAGE:
        return _unknown_result(
            "The document's vocabulary does not match any known domain "
            f"(only {coverage:.0%} of its words were seen in training)."
        )

    # Probability for each class
    clean_text = " ".join(tokens)
    probabilities = pipeline.predict_proba([clean_text])[0]
    classes = list(pipeline.classes_)
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    top_label, top_prob = ranked[0]
    prob_dict = {label: float(p) for label, p in ranked}

    # Confidence check (Unknown-document handling)
    if top_prob < threshold:
        return _unknown_result(
            f"Confidence {top_prob:.0%} is below the required threshold of {threshold:.0%}.",
            top_label=top_label,
            confidence=top_prob,
            probabilities=prob_dict,
        )

    return {
        "domain": top_label,
        "top_guess": top_label,
        "confidence": float(top_prob),
        "probabilities": prob_dict,
        "is_unknown": False,
        "reason": "",
    }


def classify_file(data, filename, threshold=None, store=True):
    """
    Full workflow for one uploaded file:
        extract text -> preprocess -> predict -> create folder -> store file
    `data` is the file content as bytes. Returns the prediction dictionary plus
    'text', 'filename' and (if store=True) 'stored_path'.
    """
    text = utils.extract_text(data, filename)
    result = predict_domain(text, threshold)
    result["text"] = text
    result["filename"] = filename
    if store:
        stored = utils.store_document(data, filename, result["domain"])
        result["stored_path"] = str(stored)
    return result


if __name__ == "__main__":
    # Command-line usage:  python predict.py my_document.pdf
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path-to-TXT|PDF|DOCX>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        output = classify_file(file_path.read_bytes(), file_path.name)
    except (ValueError, ModelNotTrainedError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"File        : {output['filename']}")
    print(f"Domain      : {output['domain']}")
    print(f"Confidence  : {output['confidence']:.1%}")
    if output["is_unknown"]:
        print(f"Reason      : {output['reason']}")
    print(f"Stored at   : {output['stored_path']}")
