import os
import time
import re
import unicodedata
import difflib
from dataclasses import dataclass
import numpy as np

from speakeasypy import Chatroom, EventType, Speakeasy

"""Run the chat bot after configuring credentials and embedding paths.

Environment variables used by the bot:

* ``SPEAKEASY_URL`` – base URL of the Speakeasy instance (required)
* ``SPEAKEASY_USERNAME`` – account username (required)
* ``SPEAKEASY_PASSWORD`` – account password (required)
* ``EMBEDDINGS_DIR`` – directory with ``entity_embeds.npy`` etc. (required)

Example::

    export SPEAKEASY_URL="https://example.org"
    export SPEAKEASY_USERNAME="MyUser"
    export SPEAKEASY_PASSWORD="MyPassword"
    export EMBEDDINGS_DIR="/path/to/embeddings"
    python chat_bot.py
"""

# --------------------------- CONFIG -------------------------------------------

CONFIG = {
    "Embeddings": {
        "EntityVec": "entity_embeds.npy",
        "EntityIds": "entity_ids.del",
        "RelationVec": "relation_embeds.npy",
        "RelationIds": "relation_ids.del",
    },
    # thresholds are easy to tune from logs
    "Thresholds": {
        "EntityLinkMin": 0.45,   # minimum fuzzy score for entity linking
        "RelationLinkMin": 0.40,  # minimum fuzzy score for relation linking
        "TopScoreMin": 0.30,     # minimum top score to accept an embedding answer
    },
    "Answer": {
        "TopKSimilar": 5,
        "TopKPrediction": 3,
    }
}


def require_env(var_name: str) -> str:
    """Fetch a required environment variable, raising a clear error if missing."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"Set the {var_name} environment variable before starting the bot.")
    return value

# ------------------------ UTILS (text & ids) ----------------------------------

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def localname(uri: str) -> str:
    """Extract local fragment or last path segment from a URI-like id."""
    if "#" in uri:
        frag = uri.rsplit("#", 1)[-1]
    else:
        frag = uri.rstrip("/").rsplit("/", 1)[-1]
    return frag

def tokenize_name(x: str) -> list[str]:
    x = normalize_text(x).lower()
    x = re.sub(r"[_\-./]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x.split()

def read_id_file(path: str) -> list[str]:
    """
    Robust reader for .del files.
    Accepts lines like:
      <uri>
      uri
      idx<tab>uri
      uri<tab>idx
    Returns: list of URI/ID strings in row order (aligned to vectors)
    """
    ids: list[str] = []
    blank_counter = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                ids.append(f"<blank:{blank_counter}>")
                blank_counter += 1
                continue
            parts = re.split(r"[\t ]+", line)
            # Heuristic: prefer the longest field that looks like a URI-ish token.
            candidate = max(parts, key=len)
            ids.append(candidate)
    return ids

def norm_for_match(s: str) -> str:
    """Lowercase, unify punctuation/whitespace and strip diacritics for matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # unify separators: colon, dash, underscore, slash → space
    s = re.sub(r"[:/_.\-]+", " ", s)
    # remove extra punctuation except alnum and space
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> list[str]:
    return norm_for_match(s).split()

def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def surface_variants(surface: str) -> list[tuple[str, list[str]]]:
    """Return a few normalized variants for a surface mention."""
    s = surface.strip()
    variants = {norm_for_match(s)}
    if ":" in s:
        variants.add(norm_for_match(s.replace(":", " ")))
        variants.add(norm_for_match(s.replace(":", "")))
    return [(v, v.split()) for v in variants if v]


def score_surface_against(
    name_norm: str,
    name_tokens: list[str],
    surface_norm: str,
    surface_tokens: list[str],
) -> float:
    """Score how well a surface form matches a stored name (0..1)."""
    jac = jaccard(name_tokens, surface_tokens)
    fuzz = difflib.SequenceMatcher(None, surface_norm, name_norm).ratio()

    boost = 0.0
    if surface_norm and name_norm:
        if name_norm.startswith(surface_norm) or surface_norm.startswith(name_norm):
            boost += 0.10
        if surface_norm in name_norm or name_norm in surface_norm:
            boost += 0.05

    score = 0.55 * fuzz + 0.40 * jac + boost
    return max(0.0, min(1.0, score))

# ---------------------- EMBEDDING INDEX / INFERENCE ---------------------------

@dataclass
class Ranked:
    uri: str
    score: float   # higher is better (cosine similarity)


@dataclass
class KGValue:
    value: str
    is_literal: bool
    lang: str | None = None

class EmbedIndex:
    def __init__(self, emb_dir: str, conf: dict):
        evec = os.path.join(emb_dir, CONFIG["Embeddings"]["EntityVec"])
        eids = os.path.join(emb_dir, CONFIG["Embeddings"]["EntityIds"])
        rvec = os.path.join(emb_dir, CONFIG["Embeddings"]["RelationVec"])
        rids = os.path.join(emb_dir, CONFIG["Embeddings"]["RelationIds"])

        # Load vectors
        print(f"Loading entity vectors from {evec} ...")
        self.E = np.load(evec)  # shape: [n_entities, d]
        print(f"Loaded entity vectors: {self.E.shape}")

        print(f"Loading relation vectors from {rvec} ...")
        self.R = np.load(rvec)  # shape: [n_relations, d]
        print(f"Loaded relation vectors: {self.R.shape}")

        # Load ids (order aligned with rows in npy files)
        print(f"Loading entity ids from {eids} ...")
        self.entity_ids = read_id_file(eids)
        print(f"Loaded entity ids: {len(self.entity_ids)}")

        print(f"Loading relation ids from {rids} ...")
        self.relation_ids = read_id_file(rids)
        print(f"Loaded relation ids: {len(self.relation_ids)}")

        # Build mappings
        self.ent2id: dict[str, int] = {self.entity_ids[i]: i for i in range(len(self.entity_ids))}
        self.id2ent: list[str] = self.entity_ids

        self.rel2id: dict[str, int] = {self.relation_ids[i]: i for i in range(len(self.relation_ids))}
        self.id2rel: list[str] = self.relation_ids

        # Precompute normalized entity matrix for cosine
        self.E = self._safe_l2norm(self.E)
        # Relations are used in raw space (TransE-style addition), then we normalize the sum
        self.thresholds = conf["Thresholds"]

        # Build a lightweight search index for entities and relations by local name tokens
        self.ent_name_index = self._build_name_index(self.entity_ids)
        self.rel_name_index = self._build_name_index(self.relation_ids)

        print("Embeddings index ready.")

    @staticmethod
    def _safe_l2norm(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    @staticmethod
    def _build_name_index(uris: list[str]) -> list[tuple[str, str, str, list[str]]]:
        # (uri, localname, norm_local, tokens_local)
        idx = []
        for u in uris:
            ln = localname(u)
            nl = norm_for_match(ln)
            idx.append((u, ln, nl, nl.split()))
        return idx

    # ---- Lookups ----

    def link_entity(self, surface: str) -> tuple[str | None, float]:
        if not surface:
            return (None, 0.0)

        # quick exact match on normalized local names
        sv = norm_for_match(surface)
        best_uri, best_score = None, 0.0

        for uri, ln, nl, toks in self.ent_name_index:
            if nl == sv:
                return (uri, 0.98)

        # try multiple surface variants and keep best
        variants = surface_variants(surface)
        for uri, ln, nl, toks in self.ent_name_index:
            for s_norm, s_toks in variants:
                sc = score_surface_against(nl, toks, s_norm, s_toks)
                if sc > best_score:
                    best_uri, best_score = uri, sc

        return (best_uri, best_score)

    def link_relation(self, phrase: str) -> tuple[str | None, float]:
        """
        Map a relation phrase like "directed by", "genre", "capital" to a relation URI
        via fuzzy localname matching.
        """
        if not phrase:
            return (None, 0.0)
        p = " ".join(tokenize_name(phrase))
        best_uri, best_score = None, 0.0
        for uri, ln, _norm_ln, _toks in self.rel_name_index:
            candidate = " ".join(tokenize_name(ln))
            r = difflib.SequenceMatcher(None, p, candidate).ratio()
            if r > best_score:
                best_uri, best_score = uri, r
        return (best_uri, best_score)
    
    def link_entity_candidates(self, surface: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k candidate URIs with scores (for clarification prompts).
        """
        if not surface:
            return []
        variants = surface_variants(surface)
        scores = np.zeros(len(self.ent_name_index), dtype=np.float32)

        for i, (uri, ln, nl, toks) in enumerate(self.ent_name_index):
            sc_max = 0.0
            for s_norm, s_toks in variants:
                sc = score_surface_against(nl, toks, s_norm, s_toks)
                if sc > sc_max:
                    sc_max = sc
            scores[i] = sc_max

        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, range(k))[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            if scores[i] <= 0.0:
                continue
            uri = self.ent_name_index[i][0]
            out.append((uri, float(scores[i])))
        return out

    # ---- Inference ----

    def similar(self, seed_uri: str, top_k: int) -> list[Ranked]:
        sid = self.ent2id.get(seed_uri)
        if sid is None:
            return []
        v = self.E[sid]  # already normalized
        sims = (self.E @ v)
        sims[sid] = -np.inf  # exclude self
        k = min(top_k, len(self.E) - 1)
        idx = np.argpartition(-sims, range(k))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Ranked(self.id2ent[i], float(sims[i])) for i in idx]

    def predict_tail(self, head_uri: str, rel_uri: str, top_k: int) -> list[Ranked]:
        hid = self.ent2id.get(head_uri)
        rid = self.rel2id.get(rel_uri)
        if hid is None or rid is None:
            return []
        v = self.E[hid] + self.R[rid]
        v_norm = v / (np.linalg.norm(v) + 1e-12)
        scores = self.E @ v_norm  # cosine to predicted tail
        k = min(top_k, len(self.E))
        idx = np.argpartition(-scores, range(k))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Ranked(self.id2ent[i], float(scores[i])) for i in idx]

def _parse_nt_line(line: str) -> tuple[str, str, str, bool, str | None] | None:
    """Parse a single N-Triples line. Returns (subj, pred, obj, is_literal, lang)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.endswith('.'):
        line = line[:-1].strip()

    n = len(line)
    i = 0

    def skip_ws(idx: int) -> int:
        while idx < n and line[idx].isspace():
            idx += 1
        return idx

    def parse_uri(idx: int) -> tuple[str, int]:
        if idx >= n or line[idx] != '<':
            raise ValueError("Expected < for URI")
        end = line.find('>', idx + 1)
        if end == -1:
            raise ValueError("Unterminated URI")
        return line[idx + 1:end], end + 1

    def parse_literal(idx: int) -> tuple[str, int, str | None]:
        if idx >= n or line[idx] != '"':
            raise ValueError("Expected quote for literal")
        idx += 1
        chars: list[str] = []
        escaped = False
        while idx < n:
            ch = line[idx]
            if ch == '"' and not escaped:
                break
            if ch == '\\' and not escaped:
                escaped = True
            else:
                escaped = False
            chars.append(ch)
            idx += 1
        if idx >= n:
            raise ValueError("Unterminated literal")
        literal = ''.join(chars)
        idx += 1
        idx = skip_ws(idx)
        lang: str | None = None
        if idx < n - 1 and line[idx] == '^' and line[idx + 1] == '^':
            idx += 2
            idx = skip_ws(idx)
            if idx < n and line[idx] == '<':
                _, idx = parse_uri(idx)
            idx = skip_ws(idx)
        elif idx < n and line[idx] == '@':
            idx += 1
            start = idx
            while idx < n and (line[idx].isalpha() or line[idx] in {'-', '_'}):
                idx += 1
            lang = line[start:idx].lower() or None
            idx = skip_ws(idx)
        return literal, idx, lang

    i = skip_ws(i)
    subj, i = parse_uri(i)
    i = skip_ws(i)
    pred, i = parse_uri(i)
    i = skip_ws(i)
    if i >= n:
        raise ValueError("Missing object")
    if line[i] == '<':
        obj, i = parse_uri(i)
        return (subj, pred, obj, False, None)
    if line[i] == '"':
        obj, i, lang = parse_literal(i)
        return (subj, pred, obj, True, lang)
    raise ValueError("Unexpected object token")


class KnowledgeGraph:
    LABEL_PREDICATES = {
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://schema.org/name",
        "http://xmlns.com/foaf/0.1/name",
        "http://purl.org/dc/terms/title",
    }

    def __init__(self, path: str):
        self.path = path
        self.adj: dict[str, dict[str, list[KGValue]]] = {}
        self.uri_labels: dict[str, list[tuple[str, str | None]]] = {}
        self.name_index: list[tuple[str, str, str, list[str]]] = []
        self._load()

    def _load(self) -> None:
        seen_name_keys: set[tuple[str, str]] = set()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    parsed = _parse_nt_line(line)
                except ValueError:
                    continue
                if not parsed:
                    continue
                subj, pred, obj, is_literal, lang = parsed
                bucket = self.adj.setdefault(subj, {}).setdefault(pred, [])
                bucket.append(KGValue(obj, is_literal, lang))
                if is_literal and pred in self.LABEL_PREDICATES:
                    self.uri_labels.setdefault(subj, []).append((obj, lang))
                    norm = norm_for_match(obj)
                    if norm:
                        key = (subj, norm)
                        if key not in seen_name_keys:
                            self.name_index.append((subj, obj, norm, norm.split()))
                            seen_name_keys.add(key)

        for subj in list(self.adj.keys()):
            ln = localname(subj)
            norm_ln = norm_for_match(ln)
            if not norm_ln:
                continue
            key = (subj, norm_ln)
            if key in seen_name_keys:
                continue
            self.name_index.append((subj, ln, norm_ln, norm_ln.split()))
            seen_name_keys.add(key)

    def link_entity(self, surface: str) -> tuple[str | None, float]:
        if not surface:
            return (None, 0.0)
        variants = surface_variants(surface)
        best_uri, best_score = None, 0.0
        for uri, _label, norm_label, toks in self.name_index:
            for s_norm, s_toks in variants:
                sc = score_surface_against(norm_label, toks, s_norm, s_toks)
                if sc > best_score:
                    best_uri, best_score = uri, sc
        return (best_uri, best_score)

    def link_entity_candidates(self, surface: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not surface:
            return []
        variants = surface_variants(surface)
        scores = np.zeros(len(self.name_index), dtype=np.float32)
        for i, (_uri, _label, norm_label, toks) in enumerate(self.name_index):
            sc_max = 0.0
            for s_norm, s_toks in variants:
                sc = score_surface_against(norm_label, toks, s_norm, s_toks)
                if sc > sc_max:
                    sc_max = sc
            scores[i] = sc_max
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, range(k))[:k]
        idx = idx[np.argsort(-scores[idx])]
        out: list[tuple[str, float]] = []
        for i in idx:
            if scores[i] <= 0.0:
                continue
            uri = self.name_index[i][0]
            out.append((uri, float(scores[i])))
        return out

    def lookup(self, subject_uri: str, relation_uri: str) -> list[KGValue]:
        return list(self.adj.get(subject_uri, {}).get(relation_uri, []))

    def best_label(self, uri: str) -> str | None:
        labels = self.uri_labels.get(uri)
        if not labels:
            return None
        for text, lang in labels:
            if lang and lang.startswith("en"):
                return text
        return labels[0][0]

    def format_value(self, value: KGValue) -> str:
        if value.is_literal:
            return value.value
        label = self.best_label(value.value)
        if label:
            return label
        return localname(value.value)


# ------------------- NLU: classify + extract mentions -------------------------

SIMILARITY_CUES = [
    r"\bsimilar (to|movies like|like)\b",
    r"\bmost similar\b", r"\bclosest\b",
    r"\brecommend(ed)?\b",
]

RELATION_HINTS = [
    # order matters: try longer/more specific first
    r"\bdirected by\b|\bdirect(ed)?\b",
    r"\bwritten by\b|\bscreenwrit(er|ten)\b",
    r"\bgenre(s)?\b",
    r"\brating\b|\bmpaa\b",
    r"\bcapital (of)?\b|\bcapital\b",
    r"\blanguage(s)?\b",
    r"\bcountry\b|\borigin\b",
]

def is_similarity(q: str) -> bool:
    qn = " ".join(q.lower().split())
    return any(re.search(pat, qn) for pat in SIMILARITY_CUES)

def extract_relation_phrase(q: str) -> str | None:
    qn = " ".join(q.lower().split())
    for pat in RELATION_HINTS:
        m = re.search(pat, qn)
        if m:
            return m.group(0)
    # also allow "what is the X of ..." → capture X
    m2 = re.search(r"what\s+is\s+the\s+([a-zA-Z\s]+?)\s+of\b", qn)
    if m2:
        return m2.group(1).strip()
    return None

def extract_entity_surface(q: str) -> str | None:
    # Prefer quoted substrings
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', q)
    for tup in quoted:
        for g in tup:
            if g:
                return g.strip()
    # Fallback: take the longest Title-like span (capitalized words)
    caps = re.findall(r"([A-Z][A-Za-z0-9&'.,:-]*(?:\s+[A-Z][A-Za-z0-9&'.,:-]*)*)", q)
    if caps:
        return max(caps, key=len).strip()
    # Final fallback: longest span of non-stopwords, even if lowercase
    tokens = re.findall(r"[A-Za-z0-9&'.,:-]+", q)
    if not tokens:
        return None
    stopwords = {
        "a", "an", "and", "are", "at", "be", "by", "can", "could", "did", "do", "does",
        "find", "for", "from", "give", "i", "in", "is", "it", "its", "list", "me",
        "movie", "movies", "of", "on", "or", "please", "recommend", "show", "similar",
        "tell", "that", "the", "their", "them", "they", "this", "those", "to", "was",
        "we", "were", "what", "when", "where", "which", "who", "why", "with", "you",
        "your", "directed", "written", "genre", "rating", "capital", "language", "country",
    }
    spans: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        t_lower = token.lower()
        if t_lower in stopwords:
            if current:
                spans.append(current)
                current = []
            continue
        current.append(token)
    if current:
        spans.append(current)
    if spans:
        spans.sort(key=lambda span: len(" ".join(span)), reverse=True)
        return " ".join(spans[0]).strip()
    # fallback to single informative token
    for token in reversed(tokens):
        if token.lower() not in stopwords:
            return token.strip()
    return None

# ----------------------- Agent (Speakeasy wiring) -----------------------------

class Agent:
    def __init__(self):
        self.url = require_env("SPEAKEASY_URL")
        self.username = require_env("SPEAKEASY_USERNAME")
        self.password = require_env("SPEAKEASY_PASSWORD")

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        emb_dir = require_env("EMBEDDINGS_DIR")
        self.index = EmbedIndex(emb_dir, CONFIG)

        kg_path = os.environ.get("KNOWLEDGE_GRAPH_PATH", "/space_mounts/atai-hs25/dataset/graph.nt")
        if os.path.exists(kg_path):
            print(f"Loading knowledge graph from {kg_path} ...")
            try:
                self.kg = KnowledgeGraph(kg_path)
                print("Knowledge graph ready.")
            except Exception as exc:
                print(f"Failed to load knowledge graph: {exc}")
                self.kg = None
        else:
            print(f"Knowledge graph file not found at {kg_path}; continuing without it.")
            self.kg = None

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        self.speakeasy.start_listening()

    # ------------------ Chat handling ------------------

    def on_new_message(self, message: str, room: Chatroom):
        print(f"New message in room {room.room_id}: {message}")
        try:
            reply = self.handle_embeddings_query(message)
            print(f"on {room.room_id}: Posting reply:\n{reply}\n")
            room.post_messages(reply)
        except Exception as e:
            err = f"Error processing your message with embeddings: {e}"
            print(f"on {room.room_id}: {err}")
            room.post_messages(err)

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom):
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    # ------------------ Embeddings logic ------------------

    def handle_embeddings_query(self, message: str) -> str:
        q = message.strip()
        if not q:
            return "Please enter a question."

        # Decide task
        sim_intent = is_similarity(q)
        rel_phrase = None if sim_intent else extract_relation_phrase(q)

        # Extract & link entity (embeddings + knowledge graph labels)
        surface = extract_entity_surface(q)
        embed_uri, el_conf = self.index.link_entity(surface) if surface else (None, 0.0)
        kg_uri, kg_conf = (self.kg.link_entity(surface) if surface and self.kg else (None, 0.0))

        threshold = CONFIG["Thresholds"]["EntityLinkMin"]
        resolved_uri: str | None = None
        has_embedding = False

        if embed_uri and el_conf >= threshold:
            resolved_uri = embed_uri
            has_embedding = True

        if not resolved_uri and kg_uri and kg_conf >= threshold:
            resolved_uri = kg_uri
            has_embedding = kg_uri in self.index.ent2id

        if not resolved_uri:
            if surface:
                suggestion_lines: list[str] = []
                seen: set[str] = set()
                for cand_uri, score in self.index.link_entity_candidates(surface, top_k=5):
                    if cand_uri in seen:
                        continue
                    seen.add(cand_uri)
                    suggestion_lines.append(f"- {localname(cand_uri)}  (score {score:.2f})")
                if self.kg:
                    for cand_uri, score in self.kg.link_entity_candidates(surface, top_k=5):
                        if cand_uri in seen:
                            continue
                        seen.add(cand_uri)
                        label = self.kg.best_label(cand_uri) or localname(cand_uri)
                        suggestion_lines.append(f"- {label}  (score {score:.2f})")
                if suggestion_lines:
                    suggestions = "\n".join(suggestion_lines)
                    return (
                        f'I could not confidently identify the entity "{surface}".\n'
                        f'Did you mean one of these?\n{suggestions}\n'
                        f'Please copy the exact name above (or paste the ID/URI if you have it).'
                    )
            hint = f' (found "{surface}" with low confidence)' if surface else ""
            return f'I could not confidently identify the entity{hint}. Please quote the exact name.'

        # Similarity
        if sim_intent:
            if not has_embedding:
                return "I found the entity but do not have embeddings for it, so I cannot compute similarity."
            topk = CONFIG["Answer"]["TopKSimilar"]
            ranked = self.index.similar(resolved_uri, topk)
            if not ranked or ranked[0].score < CONFIG["Thresholds"]["TopScoreMin"]:
                return "I couldn’t find confident similar entities."
            items = "; ".join(f"{localname(r.uri)} ({r.score:.2f})" for r in ranked)
            return f"(Embedding Answer) Similar to {surface}: {items}"

        # Link prediction
        if not rel_phrase:
            return ("I detected an embeddings answer but no relation phrase. "
                    "Try e.g. 'who directed \"MOVIE\"', 'what is the genre of \"MOVIE\"', or 'capital of \"COUNTRY\"'.")

        rel_uri, rel_conf = self.index.link_relation(rel_phrase)
        if not rel_uri or rel_conf < CONFIG["Thresholds"]["RelationLinkMin"]:
            return (
                "I detected an embeddings question but could not match the relation phrase. "
                "Please rephrase it (e.g. 'directed by', 'genre of', 'capital of')."
            )

        topk = CONFIG["Answer"]["TopKPrediction"]
        kg_items: list[str] = []
        if self.kg:
            raw_items = self.kg.lookup(resolved_uri, rel_uri)
            seen_text: set[str] = set()
            for value in raw_items:
                formatted = self.kg.format_value(value)
                if formatted in seen_text:
                    continue
                seen_text.add(formatted)
                kg_items.append(formatted)
                if len(kg_items) >= topk:
                    break
        if kg_items:
            rel_short = localname(rel_uri).replace("_", " ")
            items = "; ".join(kg_items)
            return f"(Knowledge Graph Answer) {rel_short} of {surface}: {items}"

        if not has_embedding:
            return "I couldn't find this relation in the knowledge graph and do not have embeddings for it."

        ranked = self.index.predict_tail(resolved_uri, rel_uri, topk)
        if not ranked or ranked[0].score < CONFIG["Thresholds"]["TopScoreMin"]:
            return "I couldn't produce a confident embedding prediction."

        rel_short = localname(rel_uri).replace("_", " ")
        items = "; ".join(f"{localname(r.uri)} ({r.score:.2f})" for r in ranked)
        return f"(Embedding Answer) {rel_short} of {surface}: {items}"

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

# --------------------------- ENTRYPOINT ---------------------------------------

if __name__ == '__main__':
    demo_bot = Agent()
    demo_bot.listen()