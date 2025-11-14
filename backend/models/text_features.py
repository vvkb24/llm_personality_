"""
Text feature extraction module.

Provides:
- sentence_transformer embedding (SBERT)
- lexical & syntactic features:
  - pronoun ratio
  - average word length
  - avg sentence length
  - function word ratio
  - negative/positive sentiment counts (simple lexicon)
  - POS tag ratios (noun/verb/adj/adv)
- a single function `text_to_features(text)` that returns:
  (embedding_vector, lexical_feature_dict)

Detailed comments included for learning purposes.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
import string
import os

# Ensure nltk data present
nltk_data_dir = os.path.join(os.path.dirname(__file__), "..", "nltk_data")
# Use NLTK downloader once. If your environment blocks downloads, run them manually.
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

STOPWORDS = set(stopwords.words('english'))
PUNCT = set(string.punctuation)

# Small sentiment lexicons for quick features (not comprehensive)
POS_LEXICON = {"good", "great", "happy", "joy", "love", "excellent", "wonderful", "positive", "awesome", "amazing"}
NEG_LEXICON = {"bad", "sad", "terrible", "hate", "awful", "horrible", "negative", "angry", "upset", "worse"}

# Load SBERT model (compact & fast: all-MiniLM-L6-v2)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_sbert = None

def get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(SBERT_MODEL_NAME)
    return _sbert

def safe_tokenize(text: str):
    """Tokenize and normalize text; returns list of tokens (words)."""
    if not text:
        return []
    tokens = word_tokenize(text)
    # lowercase and filter punctuation
    tokens = [t.lower() for t in tokens if t not in PUNCT]
    return tokens

def lexical_features(text: str):
    """
    Compute simple lexical and syntactic features.
    Returns a dict of features.
    """
    sents = sent_tokenize(text) if text else []
    tokens = safe_tokenize(text)
    token_count = len(tokens)
    sent_count = len(sents) if sents else 1

    # Pronouns (subject & object) via POS tags (PRP, PRP$)
    pos_tags = pos_tag(tokens) if tokens else []
    pronouns = sum(1 for _, tag in pos_tags if tag in ("PRP", "PRP$"))

    # POS distributions: noun, verb, adjective, adverb
    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    adj_tags = {"JJ", "JJR", "JJS"}
    adv_tags = {"RB", "RBR", "RBS"}

    noun_count = sum(1 for _, t in pos_tags if t in noun_tags)
    verb_count = sum(1 for _, t in pos_tags if t in verb_tags)
    adj_count = sum(1 for _, t in pos_tags if t in adj_tags)
    adv_count = sum(1 for _, t in pos_tags if t in adv_tags)

    # Average word length
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
    # Average sentence length in words
    avg_sent_len = token_count / sent_count if sent_count > 0 else 0.0

    # Function word ratio (simple proxy: stopwords)
    func_word_ratio = sum(1 for t in tokens if t in STOPWORDS) / (token_count if token_count>0 else 1)

    # Sentiment lexicon counts
    pos_count = sum(1 for t in tokens if t in POS_LEXICON)
    neg_count = sum(1 for t in tokens if t in NEG_LEXICON)

    features = {
        "token_count": token_count,
        "avg_word_len": float(avg_word_len),
        "avg_sent_len": float(avg_sent_len),
        "pronoun_ratio": float(pronouns / (token_count if token_count>0 else 1)),
        "func_word_ratio": float(func_word_ratio),
        "noun_ratio": float(noun_count / (token_count if token_count>0 else 1)),
        "verb_ratio": float(verb_count / (token_count if token_count>0 else 1)),
        "adj_ratio": float(adj_count / (token_count if token_count>0 else 1)),
        "adv_ratio": float(adv_count / (token_count if token_count>0 else 1)),
        "pos_lex_count": int(pos_count),
        "neg_lex_count": int(neg_count),
    }
    return features

def text_embedding(text: str):
    """Return SBERT embedding as numpy array (dtype float32)."""
    model = get_sbert()
    emb = model.encode(text)
    return emb.astype(np.float32)

def text_to_features(text: str):
    """
    High-level function that returns:
    - embedding: numpy array (d,)
    - lex_features: dict of numerical features
    """
    emb = text_embedding(text)
    lex = lexical_features(text)
    return emb, lex
