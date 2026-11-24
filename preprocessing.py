"""
preprocessing.py
================
NLP preprocessing pipeline used by BOTH training and prediction.

Using the exact same cleaning code in both places is critical: if the text is
cleaned differently at prediction time, the model sees data that looks
different from what it was trained on and accuracy drops.

Pipeline (in order):
    1. Lowercasing
    2. URL / e-mail removal and special-character removal
    3. Tokenization            (NLTK RegexpTokenizer - needs no downloads)
    4. Stopword removal        (NLTK English stopwords)
    5. Lemmatization           (NLTK WordNetLemmatizer)

Robustness: NLTK data (stopwords, wordnet) is downloaded automatically the
first time. If the download is impossible (no internet), the code falls back
to a built-in stopword list and skips lemmatization instead of crashing.
"""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# --------------------------------------------------------------------------
# Fallback stopwords (used only if the NLTK stopword corpus is unavailable)
# --------------------------------------------------------------------------
_FALLBACK_STOPWORDS = set(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can did do does doing down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with you your yours yourself yourselves
    """.split()
)

# Compiled regular expressions (compiled once = faster)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")

# Tokenizer that keeps only alphabetic words (drops digits and punctuation)
_TOKENIZER = RegexpTokenizer(r"[a-z]+")


@lru_cache(maxsize=1)
def get_stopwords():
    """Return the English stopword set (downloads the NLTK corpus if needed)."""
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        pass
    try:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))
    except Exception:
        print("[preprocessing] NLTK stopwords unavailable - using built-in list.")
        return frozenset(_FALLBACK_STOPWORDS)


@lru_cache(maxsize=1)
def get_lemmatizer():
    """Return a WordNetLemmatizer, or None if WordNet cannot be loaded."""
    lemmatizer = WordNetLemmatizer()
    for _ in range(2):  # attempt 1: already installed, attempt 2: after download
        try:
            lemmatizer.lemmatize("tests")  # probe call: raises LookupError if missing
            return lemmatizer
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                break
    print("[preprocessing] NLTK WordNet unavailable - lemmatization skipped.")
    return None


def preprocess_tokens(text):
    """Convert raw text into a list of clean, lemmatized tokens."""
    if not isinstance(text, str):
        return []

    # Step 1: lowercasing
    text = text.lower()

    # Step 2: remove URLs and e-mail addresses (they carry no domain meaning)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)

    # Step 3: tokenization (letters only, so numbers/punctuation disappear)
    tokens = _TOKENIZER.tokenize(text)

    # Step 4: stopword removal + drop 1-letter tokens
    stop_words = get_stopwords()
    tokens = [t for t in tokens if len(t) >= 2 and t not in stop_words]

    # Step 5: lemmatization (noun pass, then verb pass: "studies" -> "study")
    lemmatizer = get_lemmatizer()
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="n"), pos="v") for t in tokens]

    return tokens


def preprocess_text(text):
    """Return the cleaned text as ONE string (what TF-IDF expects)."""
    return " ".join(preprocess_tokens(text))


if __name__ == "__main__":
    # Quick self-test:  python preprocessing.py
    sample = "The doctors were treating patients' infections with new antibiotics! Visit www.example.com"
    print("Original :", sample)
    print("Cleaned  :", preprocess_text(sample))
