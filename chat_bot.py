import os
import time
import re
import unicodedata
import difflib
from dataclasses import dataclass
import numpy as np

from speakeasypy import Chatroom, EventType, Speakeasy

"""
To run the bot do the following:
1.  Open terminal
2.  Run "python chat_bot.py"
3.  Wait until graph is loaded and the bot is listening (this can take a while)


To test and interact with the bot do the following:

1.  Go to https://speakeasy.ifi.uzh.ch/
2.  Login
3.  Go to "Chat"
4.  Click on "Request Chat"
5.  Enter "CyanPeekingMouse" and click on "Request"
"""

# --------------------------- CONFIG -------------------------------------------

CONFIG = {
    "Hosting": {
        "URL": "https://speakeasy.ifi.uzh.ch",
        "Username": "CyanPeekingMouse",
        "Password": "Qe5Hf3zJ",
    },
    "Embeddings": {
        "Dir": "/space_mounts/atai-hs25/dataset/embeddings",
        "EntityVec": "entity_embeds.npy",
        "EntityIds": "entity_ids.del",
        "RelationVec": "relation_embeds.npy",
        "RelationIds": "relation_ids.del",
    },
    # thresholds are easy to tune from logs
    "Thresholds": {
        "EntityLinkMin": 0.45,   # minimum fuzzy score for entity linking
        "TopScoreMin": 0.30,     # minimum top score to accept an embedding answer
    },
    "Answer": {
        "TopKSimilar": 5,
        "TopKPrediction": 3,
    }
}

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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                ids.append("")
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

# ---------------------- EMBEDDING INDEX / INFERENCE ---------------------------

@dataclass
class Ranked:
    uri: str
    score: float   # higher is better (cosine similarity)

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
        variants = self._surface_variants(surface)
        for uri, ln, nl, toks in self.ent_name_index:
            for s_norm, s_toks in variants:
                sc = self._score_surface_against(nl, toks, s_norm, s_toks)
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
        for uri, ln, _ in self.rel_name_index:
            r = difflib.SequenceMatcher(None, p, " ".join(tokenize_name(ln))).ratio()
            if r > best_score:
                best_uri, best_score = uri, r
        return (best_uri, best_score)
    
    def _surface_variants(self, surface: str) -> list[tuple[str, list[str]]]:
        """
        Build a few normalized variants to be more tolerant with punctuation/underscores/colons.
        Returns list of (normalized_string, tokens).
        """
        s = surface.strip()
        variants = {norm_for_match(s)}
        # also try replacing ":" with space and with nothing (users often include subtitles)
        variants.add(norm_for_match(s.replace(":", " ")))
        variants.add(norm_for_match(s.replace(":", "")))
        # underscores, dashes and slashes are handled in norm_for_match already
        return [(v, v.split()) for v in variants if v]

    def _score_surface_against(self, name_norm: str, name_tokens: list[str], surface_norm: str, surface_tokens: list[str]) -> float:
        """
        Combine token Jaccard + fuzzy ratio + substring/startswith boosts into a single 0..1 score.
        """
        jac = jaccard(name_tokens, surface_tokens)
        # difflib on normalized single strings
        fuzz = difflib.SequenceMatcher(None, surface_norm, name_norm).ratio()

        boost = 0.0
        # substring / startswith help when titles include punctuation or subtitles
        if surface_norm and name_norm:
            if name_norm.startswith(surface_norm) or surface_norm.startswith(name_norm):
                boost += 0.10
            if surface_norm in name_norm or name_norm in surface_norm:
                boost += 0.05

        # weighted sum (tune if needed)
        score = 0.55 * fuzz + 0.40 * jac + boost
        return max(0.0, min(1.0, score))

    def link_entity_candidates(self, surface: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k candidate URIs with scores (for clarification prompts).
        """
        if not surface:
            return []
        variants = self._surface_variants(surface)
        scores = np.zeros(len(self.ent_name_index), dtype=np.float32)

        for i, (uri, ln, nl, toks) in enumerate(self.ent_name_index):
            sc_max = 0.0
            for s_norm, s_toks in variants:
                sc = self._score_surface_against(nl, toks, s_norm, s_toks)
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
    return None

# ----------------------- Agent (Speakeasy wiring) -----------------------------

class Agent:
    def __init__(self):
        self.url = CONFIG["Hosting"]["URL"]
        self.username = CONFIG["Hosting"]["Username"]
        self.password = CONFIG["Hosting"]["Password"]

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        emb_dir = CONFIG["Embeddings"]["Dir"]
        self.index = EmbedIndex(emb_dir, CONFIG)

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

        # Extract & link entity
        surface = extract_entity_surface(q)
        uri, el_conf = self.index.link_entity(surface) if surface else (None, 0.0)
        if not uri or el_conf < CONFIG["Thresholds"]["EntityLinkMin"]:
            hint = f' (found "{surface}" with low confidence)' if surface else ""
            return f'I could not confidently identify the entity{hint}. Please quote the exact name.'

        # Similarity
        if sim_intent:
            topk = CONFIG["Answer"]["TopKSimilar"]
            ranked = self.index.similar(uri, topk)
            if not ranked or ranked[0].score < CONFIG["Thresholds"]["TopScoreMin"]:
                return "I couldn’t find confident similar entities."
            items = "; ".join(f"{localname(r.uri)} ({r.score:.2f})" for r in ranked)
            return f"(Embedding Answer) Similar to {surface}: {items}"

        # Link prediction
        if not rel_phrase:
            return ("I detected an embeddings answer but no relation phrase. "
                    "Try e.g. 'who directed \"MOVIE\"', 'what is the genre of \"MOVIE\"', or 'capital of \"COUNTRY\"'.")

        rel_uri, rel_conf = self.index.link_relation(rel_phrase)
        if not uri or el_conf < CONFIG["Thresholds"]["EntityLinkMin"]:
            if surface:
                cand = self.index.link_entity_candidates(surface, top_k=5)
                if cand:
                    suggestions = "\n".join(
                        f"- {localname(u)}  (score {s:.2f})" for u, s in cand
                    )
                    return (
                        f'I could not confidently identify the entity "{surface}".\n'
                        f'Did you mean one of these?\n{suggestions}\n'
                        f'Please copy the exact name above (or paste the ID/URI if you have it).'
                    )
            return "I could not confidently identify the entity. Please quote the exact name."

        topk = CONFIG["Answer"]["TopKPrediction"]
        ranked = self.index.predict_tail(uri, rel_uri, topk)
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