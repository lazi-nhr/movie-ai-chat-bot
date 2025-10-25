"""Text processing utilities."""
import re
import unicodedata
import difflib

def normalize_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    # Convert to lowercase and normalize unicode
    text = unicodedata.normalize("NFKD", text.lower())
    # Remove diacritics
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Replace special chars with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()

def extract_entity(text: str) -> str | None:
    """Extract potential entity mention from text."""
    # Try quoted text first
    quoted = re.findall(r'"([^"]+)"', text)
    for match in quoted:
        for group in match:
            if group:
                return group.strip()
    
    # Try capitalized sequences
    caps = re.findall(r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)", text)
    if caps:
        return max(caps, key=len).strip()
    
    return None

def get_similarity_score(text1: str, text2: str) -> float:
    """Get similarity score between two texts."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 or not norm2:
        return 0.0
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()

def extract_relation(text: str) -> str | None:
    """Extract relation phrase from text."""
    text = text.lower()
    patterns = [
        (r"directed by|director", "directed_by"),
        (r"genre", "genre"),
        (r"rating|mpaa", "rating"),
        (r"capital of|capital", "capital"),
        (r"language", "language"),
        (r"country|origin", "country"),
    ]
    
    for pattern, relation in patterns:
        if re.search(pattern, text):
            return relation
            
    return None