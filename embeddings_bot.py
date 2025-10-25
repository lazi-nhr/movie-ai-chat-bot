import os
import re
import time
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from rdflib import Graph, Namespace
from speakeasypy import Chatroom, EventType, Speakeasy

# -----------------------------
# Instructions
# -----------------------------
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

# -----------------------------
# Dataset structure
# -----------------------------
"""
The dataset directory structure is as follows:

/space_mounts/atai-hs25/dataset/
├── additional/
│   ├── images.json
│   ├── movie_plots.csv
│   ├── plots.csv
│   └── user_comments.csv
│
├── embeddings/
│   ├── entity_embeds.npy
│   ├── entity_ids.del
│   ├── relation_embeds.npy
│   └── relation_ids.del
│
├── graph.nt
├── graph.tsv
│
├── image_features/
│   ├── 0000.pkl
│   ├── 0001.pkl
│   ├──  ... 
│   └── 0347.pkl
│
├── images/
│   ├── 0000/
│   ├── 0001/
│   ├──  ... 
│   └── 0347/
│
└── ratings/
    ├── item_ratings.csv
    └── user_ratings.csv
"""

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    "Hosting": {
        "URL": "https://speakeasy.ifi.uzh.ch",
        "Username": "CyanPeekingMouse",
        "Password": "Qe5Hf3zJ"
    },
    "Data": {
        "Directory": "/space_mounts/atai-hs25/dataset",
        "EntityEmbeds": "embeddings/entity_embeds.npy",
        "EntityIds": "embeddings/entity_ids.del",
        "RelationEmbeds": "embeddings/relation_embeds.npy",
        "RelationIds": "embeddings/relation_ids.del",
        # Optional: use labels from the RDF graph for nicer names in answers
        "RDF": "graph.nt",
    },
    "Answering": {
        "TopK": 3,          # how many suggestions to show
        "MaxCandidates": 0, # 0 = all entities; else sample N for speed on huge graphs
    }
}

# -----------------------------
# Utilities
# -----------------------------
def load_list_file(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load a .del / .txt style file where each line is either:
      - 'ID'           (single column), or
      - 'ID<TAB>label' (two columns). We keep the first column as the canonical ID,
        but return label if available for nicer relation/entity names fallback.
    We primarily return the first column list; a side dict of optional labels can be built by caller.
    
    Args:
        path (str): Path to the file to load
        
    Returns:
        Tuple[List[str], Dict[str, str]]: List of IDs and dictionary mapping IDs to labels
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there are issues reading the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    ids = []
    # Also return a mapping of ID -> optional human label if present
    labels_by_id: Dict[str, str] = {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 1:
                    ids.append(parts[0])
                else:
                    ids.append(parts[0])
                    # prefer non-empty label
                    if parts[1].strip():
                        labels_by_id[parts[0]] = parts[1].strip()
    except IOError as e:
        logger.error(f"Error reading file {path}: {e}")
        raise
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                ids.append(parts[0])
            else:
                ids.append(parts[0])
                # prefer non-empty label
                if parts[1].strip():
                    labels_by_id[parts[0]] = parts[1].strip()
    return ids, labels_by_id

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=2))

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def best_fuzzy_match(query: str, candidates: Dict[str, str], min_ratio: float = 0.65) -> Optional[str]:
    """
    Very lightweight fuzzy matcher without external deps:
    - Convert both sides to sets of tokens
    - Jaccard similarity over tokens + a simple subsequence bonus
    Returns the candidate key with the best score if above threshold.
    """
    q = normalize_text(query)
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    best_key, best_score = None, 0.0
    for key, label in candidates.items():
        t = normalize_text(label)
        t_tokens = set(re.findall(r"[a-z0-9]+", t))
        # Jaccard over tokens
        inter = len(q_tokens & t_tokens)
        union = max(1, len(q_tokens | t_tokens))
        jacc = inter / union
        # subsequence-ish bonus
        bonus = 0.15 if all(tok in t for tok in q_tokens) else 0.0
        score = jacc + bonus
        if score > best_score:
            best_key, best_score = key, score
    return best_key if best_score >= min_ratio else None

# -----------------------------
# Embedding-based QA core
# -----------------------------
class EmbeddingKB:
    def __init__(self, data_dir: str, topk: int = 3, max_candidates: int = 0):
        self.data_dir = data_dir
        self.topk = topk
        self.max_candidates = max_candidates

        # Load embeddings
        ent_path = os.path.join(data_dir, CONFIG["Data"]["EntityEmbeds"])
        rel_path = os.path.join(data_dir, CONFIG["Data"]["RelationEmbeds"])
        self.E = np.load(ent_path)  # shape: [N_entities, d]
        self.R = np.load(rel_path)  # shape: [N_relations, d]

        # Load id lists
        ent_ids_path = os.path.join(data_dir, CONFIG["Data"]["EntityIds"])
        rel_ids_path = os.path.join(data_dir, CONFIG["Data"]["RelationIds"])
        self.entity_ids, self.entity_optional_labels = load_list_file(ent_ids_path)
        self.relation_ids, self.relation_optional_labels = load_list_file(rel_ids_path)

        # Build index mappings
        self.ent2idx: Dict[str, int] = {eid: i for i, eid in enumerate(self.entity_ids)}
        self.rel2idx: Dict[str, int] = {rid: i for i, rid in enumerate(self.relation_ids)}

        # Optionally load RDF graph to resolve pretty labels
        self.label_graph = Graph()
        nt_path = os.path.join(data_dir, CONFIG["Data"]["RDF"])
        if os.path.exists(nt_path):
            try:
                logger.info(f"Loading RDF graph from {nt_path}")
                self.label_graph.parse(nt_path, format="nt")
                logger.info(f"Successfully loaded RDF graph with {len(self.label_graph)} triples")
            except Exception as e:
                logger.error(f"Failed to load RDF graph from {nt_path}: {e}")
                # Create empty graph to allow fallback behavior
                self.label_graph = Graph()

        # Fast label caches
        self.entity_labels_cache: Dict[str, str] = {}
        self.relation_labels_cache: Dict[str, str] = {}

        # Precompute a minimal alias map for common relations (extend as needed)
        # Keys are lowercase patterns to search in questions; values will be resolved to the best relation ID.
        self.RELATION_ALIASES = {
            "screenwriter": ["screenwriter", "written by", "writer", "screenplay"],
            "mpaa film rating": ["mpaa", "mpaa rating", "film rating", "mpaa film rating", "rated"],
            "genre": ["genre", "type of film", "category"],
            # Add more aliases here if your dataset includes them (director, cast, producer, country, language, etc.)
        }

        # Build a relation name dictionary we can fuzzy match against (using any available labels or the ID itself)
        self.relation_name_by_id: Dict[str, str] = {}
        for rid in self.relation_ids:
            label = self.relation_optional_labels.get(rid) or rid
            self.relation_name_by_id[rid] = label

    # -------------------------
    # Label resolution helpers
    # -------------------------
    def label_for_entity(self, ent_id: str) -> str:
        if ent_id in self.entity_labels_cache:
            return self.entity_labels_cache[ent_id]

        # Prefer optional labels shipped in relation/entity_ids files
        if ent_id in self.entity_optional_labels:
            label = self.entity_optional_labels[ent_id]
            self.entity_labels_cache[ent_id] = label
            return label

        # Try RDF label predicates
        if len(self.label_graph):
            RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
            SCHEMA = Namespace("http://schema.org/")
            WD = Namespace("http://www.wikidata.org/entity/")
            subj = WD[ent_id] if not ent_id.startswith("http") else ent_id

            # Collect a few common label predicates
            for p in (RDFS.label, SCHEMA.name):
                try:
                    for o in self.label_graph.objects(subj, p):
                        label = str(o)
                        if label:
                            self.entity_labels_cache[ent_id] = label
                            return label
                except Exception:
                    pass

        # Fallback: show the ID itself
        self.entity_labels_cache[ent_id] = ent_id
        return ent_id

    def label_for_relation(self, rel_id: str) -> str:
        if rel_id in self.relation_labels_cache:
            return self.relation_labels_cache[rel_id]
        label = self.relation_optional_labels.get(rel_id) or self.relation_name_by_id.get(rel_id) or rel_id
        self.relation_labels_cache[rel_id] = label
        return label

    # -------------------------
    # Parsing question -> (head entity, relation)
    # -------------------------
    def detect_head_entity(self, question: str) -> Optional[str]:
        """
        Try to pull a film/person/title mention out of the question by matching against known labels or IDs.
        Strategy:
          1) If question contains a quoted span " ... ", try that first.
          2) Fuzzy match the full question against known entity labels (cheap token-Jaccard).
          3) As a last resort, look for Wikidata-like IDs Qxxxx in the text.
        """
        q = question.strip()

        # 1) Quoted span
        m = re.search(r"['\"]([^'\"]{2,})['\"]", q)
        if m:
            title = m.group(1)
            # Build a {entity_id: label} dict for fuzzy match
            label_map: Dict[str, str] = {}
            # We only have optional labels for some entities; fall back to ID itself
            for eid in self.entity_ids:
                label_map[eid] = self.entity_optional_labels.get(eid, eid)
            best = best_fuzzy_match(title, label_map, min_ratio=0.55)
            if best:
                return best

        # 2) Fuzzy match the whole question against labels
        label_map = {eid: self.entity_optional_labels.get(eid, eid) for eid in self.entity_ids}
        best = best_fuzzy_match(q, label_map, min_ratio=0.58)
        if best:
            return best

        # 3) Raw QID in text
        m2 = re.search(r"\b(Q\d+)\b", q, flags=re.IGNORECASE)
        if m2:
            qid = m2.group(1)
            if qid in self.ent2idx:
                return qid

        return None

    def detect_relation(self, question: str) -> Optional[str]:
        """
        Map the question to a relation embedding index by:
          a) checking aliases -> fuzzy match into relation list
          b) fuzzy matching the question directly into relation names if (a) fails
        """
        qn = normalize_text(question)

        # a) alias detection
        for canonical, variants in self.RELATION_ALIASES.items():
            if any(v in qn for v in variants):
                # fuzzy into relation list using the canonical alias as the query (or the longest variant)
                query = canonical
                # Build dict {rel_id: readable-name}
                match = best_fuzzy_match(query, self.relation_name_by_id, min_ratio=0.45)
                if match:
                    return match

        # b) direct fuzzy against relation names
        match = best_fuzzy_match(qn, self.relation_name_by_id, min_ratio=0.55)
        return match

    # -------------------------
    # TransE scoring
    # -------------------------
    def predict_tails_transe(self, head_id: str, rel_id: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Given head entity and relation, return top-k tails with smallest ||h + r - t||_2.
        """
        hi = self.ent2idx[head_id]
        ri = self.rel2idx[rel_id]
        h = self.E[hi]
        r = self.R[ri]

        # Optional candidate sampling for speed on very large graphs
        if self.max_candidates and self.max_candidates < len(self.entity_ids):
            cand_idx = np.random.choice(len(self.entity_ids), size=self.max_candidates, replace=False)
            T = self.E[cand_idx]
            dists = np.linalg.norm((h + r)[None, :] - T, axis=1)
            topk_idx_local = np.argpartition(dists, kth=min(k, len(dists)-1))[:k]
            pairs = sorted([(self.entity_ids[int(cand_idx[i])], float(dists[i])) for i in topk_idx_local],
                           key=lambda x: x[1])[:k]
            return pairs

        # Full scan
        # (For speed in production, use Faiss or ANN. Here, numpy is fine.)
        diffs = self.E - (h + r)
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        topk_idx = np.argpartition(dists, kth=min(k, len(dists)-1))[:k]
        pairs = sorted([(self.entity_ids[int(i)], float(dists[int(i)])) for i in topk_idx], key=lambda x: x[1])[:k]
        return pairs

    # -------------------------
    # Public API
    # -------------------------
    def answer(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            return "Please provide a non-empty question."
            
        question = question.strip()
        logger.info(f"Processing question: {question}")
            
        head_id = self.detect_head_entity(question)
        if not head_id:
            logger.info("No head entity detected in question")
            return "I couldn't identify the subject entity in your question from the embeddings index."

        rel_id = self.detect_relation(question)
        if not rel_id:
            return "I couldn't map your question to a known relation in the embeddings."

        # Predict
        k = CONFIG["Answering"]["TopK"]
        preds = self.predict_tails_transe(head_id, rel_id, k=k)

        # Format answer
        labels = [self.label_for_entity(eid) for eid, _ in preds]
        # Basic de-duplication on labels
        seen = set()
        uniq_labels = []
        for lab in labels:
            if lab not in seen:
                uniq_labels.append(lab)
                seen.add(lab)

        # Keep the exact phrasing style default: "<The answer suggested by embeddings: A, B, and C.>"
        if len(uniq_labels) == 0:
            return "No suggestion found from embeddings."

        if len(uniq_labels) == 1:
            joined = uniq_labels[0]
        elif len(uniq_labels) == 2:
            joined = f"{uniq_labels[0]} and {uniq_labels[1]}"
        else:
            joined = ", ".join(uniq_labels[:-1]) + f", and {uniq_labels[-1]}"

        return f"The answer suggested by embeddings: {joined}."

# -----------------------------
# Speakeasy Agent (identical wiring)
# -----------------------------
class EmbeddingsAgent:
    def __init__(self):
        # Validate required configuration
        if not CONFIG["Hosting"]["Username"] or not CONFIG["Hosting"]["Password"]:
            raise ValueError("Missing required environment variables: SPEAKEASY_USERNAME and/or SPEAKEASY_PASSWORD")
            
        if not os.path.exists(CONFIG["Data"]["Directory"]):
            raise ValueError(f"Data directory does not exist: {CONFIG['Data']['Directory']}")
            
        self.url = CONFIG["Hosting"]["URL"]
        self.username = CONFIG["Hosting"]["Username"]
        self.password = CONFIG["Hosting"]["Password"]
        self.data_dir = CONFIG["Data"]["Directory"]
        
        logger.info(f"Initializing EmbeddingsAgent with data directory: {self.data_dir}")

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        self.kb = EmbeddingKB(
            data_dir=self.data_dir,
            topk=CONFIG["Answering"]["TopK"],
            max_candidates=CONFIG["Answering"]["MaxCandidates"],
        )

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        logger.info(f"New message in room {room.room_id}: {message}")
        try:
            # Accept plain English questions and answer from embeddings
            reply = self.kb.answer(message)
            logger.info(f"Room {room.room_id}: Posting reply:\n{reply}")
            room.post_messages(reply)
        except Exception as e:
            reply = f"Error processing your question with embeddings: {e}"
            logger.error(f"Room {room.room_id}: Error processing message: {e}", exc_info=True)
            room.post_messages(reply)

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom):
        logger.info(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    bot = EmbeddingsAgent()
    bot.listen()