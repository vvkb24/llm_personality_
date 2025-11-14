import pytest
from backend.models.text_features import text_to_features

def test_text_to_features_nonempty():
    emb, lex = text_to_features("I love programming and solving problems.")
    assert emb is not None
    assert len(emb) > 0
    assert isinstance(lex, dict)
    assert "token_count" in lex

def test_text_to_features_empty():
    emb, lex = text_to_features("")
    assert emb is not None
    assert len(emb) > 0
    assert lex["token_count"] == 0
