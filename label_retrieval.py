import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

from rdflib import Graph, URIRef, Literal

# --------------------------- ID FILE PARSING ----------------------------------

QID_URI_PREFIX = "http://www.wikidata.org/entity/"
PID_URI_PREFIX = "http://www.wikidata.org/prop/direct/"

QID_RE = re.compile(r"^(?:https?://www\.wikidata\.org/entity/)?(Q\d+)$", re.I)
PID_RE = re.compile(r"^(?:https?://www\.wikidata\.org/prop/direct/)?(P\d+)$", re.I)

def normalize_entity_id(
        token: str
        ) -> str | None:
    """
    Accepts bare QID or full entity URI; returns full Wikidata entity URI.
    """
    tok = token.strip()
    m = QID_RE.match(tok)
    if m:
        return f"{QID_URI_PREFIX}{m.group(1)}"
    return None

def normalize_relation_id(
        token: str
        ) -> str | None:
    """
    Accepts bare PID or full direct property URI; returns full Wikidata direct property URI.
    """
    tok = token.strip()
    m = PID_RE.match(tok)
    if m:
        return f"{PID_URI_PREFIX}{m.group(1)}"
    return None

def read_ids_file(
        path: str, 
        kind: str
        ) -> list[str]:
    """
    Reads a .del-like file that may contain:
      - a single token per line (URI or bare ID)
      - two tokens per line (idx + URI) or (URI + idx)
    Returns a list of normalized URIs (full Wikidata URIs), preserving order.
    'kind' ∈ {'entity','relation'} decides which normalizer to apply.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                out.append("")
                continue
            parts = re.split(r"[\t ]+", line)
            # prefer tokens that look like QIDs/PIDs or URIs
            best = None
            for t in parts:
                if kind == "entity":
                    cand = normalize_entity_id(t) or (normalize_entity_id(extract_qid_from_uri(t)) if t.startswith("http") else None)
                else:
                    cand = normalize_relation_id(t) or (normalize_relation_id(extract_pid_from_uri(t)) if t.startswith("http") else None)
                if cand:
                    best = cand
                    break
            # fallback: try last token
            if best is None:
                t = parts[-1]
                best = normalize_entity_id(t) if kind == "entity" else normalize_relation_id(t)
                if best is None and t.startswith("http"):
                    # as a last resort, accept the URI as-is (may still be correct)
                    best = t
            out.append(best or "")
    return out

def extract_qid_from_uri(
        uri: str
        ) -> str:
    # return bare QID if present, else original string
    m = re.search(r"(Q\d+)", uri)
    return m.group(1) if m else uri

def extract_pid_from_uri(
        uri: str
        ) -> str:
    # return bare PID if present, else original string
    m = re.search(r"(P\d+)", uri)
    return m.group(1) if m else uri

# ------------------------------ LABELS ----------------------------------------

LABEL_PREDICATES = {
    str(URIRef("http://www.w3.org/2000/01/rdf-schema#label")),   # rdfs:label
    str(URIRef("http://www.w3.org/2004/02/skos/core#altLabel")), # skos:altLabel
    str(URIRef("http://schema.org/name")),                       # schema:name
    str(URIRef("http://xmlns.com/foaf/0.1/name")),               # foaf:name
}

def norm_text(
        s: str
        ) -> str:
    """
    Lowercase, strip diacritics, normalize separators, keep letters/numbers/spaces.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[:/_\-\.]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def collect_labels(
        graph_path: str,
        keep_entities: set[str],
        keep_relations: set[str],
        langs: tuple[str, ...]
        ) -> tuple[dict, dict, list[tuple[str,str,str,str]]]:
    """
    Iterate over graph.nt and collect labels for:
      - entities in keep_entities
      - relations in keep_relations
    Returns:
      forward_map: uri -> {'labels': set([...]), 'norms': set([...])}
      inverted_index: norm_text -> set([uri, ...])
      rows_for_csv: list of (uri, kind, label, lang)
    """
    g = Graph()
    print(f"Parsing graph: {graph_path}")
    g.parse(graph_path, format="nt")
    print(f"Triples loaded: {len(g)}")

    forward: dict[str, dict] = {}
    inverted: dict[str, set] = defaultdict(set)
    rows: list[tuple[str,str,str,str]] = []

    def add(uri: str, kind: str, label: str, lang: str):
        # nt = norm_text(label)
        nt = label
        if not nt:
            return
        if uri not in forward:
            forward[uri] = {"labels": set(), "norms": set()}
        forward[uri]["labels"].add(label)
        forward[uri]["norms"].add(nt)
        inverted[nt].add(uri)
        rows.append((uri, kind, label, lang or ""))

    for s, p, o in g.triples((None, None, None)):
        if not isinstance(s, URIRef) or not isinstance(p, URIRef):
            continue
        pred = str(p)
        if pred not in LABEL_PREDICATES:
            continue
        if not isinstance(o, Literal):
            continue
        lang = o.language or ""
        if lang not in langs:
            continue
        subj = str(s)

        if subj in keep_entities:
            add(subj, "entity", str(o), lang)
        elif subj in keep_relations:
            add(subj, "relation", str(o), lang)

    return forward, inverted, rows

# ------------------------------ WRITING ---------------------------------------

def write_csv(
        rows: list[tuple[str,str,str,str]], 
        out_dir: str
        ):
    # Split rows by kind (entity/relation)
    entity_rows = [row for row in rows if row[1] == "entity"]
    relation_rows = [row for row in rows if row[1] == "relation"]
    
    # Write entities CSV
    entities_csv = os.path.join(out_dir, "entities", "entity_labels.csv")
    os.makedirs(os.path.dirname(entities_csv), exist_ok=True)
    with open(entities_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uri", "kind", "label", "lang", "type"])
        for row in entity_rows:
            # Add type as "entity"
            w.writerow(list(row) + ["entity"])
    print(f"Wrote {entities_csv} ({len(entity_rows)} rows)")
    
    # Write relations CSV
    relations_csv = os.path.join(out_dir, "relations", "relation_labels.csv")
    os.makedirs(os.path.dirname(relations_csv), exist_ok=True)
    with open(relations_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uri", "kind", "label", "lang", "type"])
        for row in relation_rows:
            # Add type as "relation"
            w.writerow(list(row) + ["relation"])
    print(f"Wrote {relations_csv} ({len(relation_rows)} rows)")

def write_json(
        obj, 
        path: str
        ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, dict):
            # Write each entry on a new line
            f.write("{\n")
            items = list(obj.items())
            for i, (key, value) in enumerate(items):
                json_str = json.dumps({key: value}, ensure_ascii=False)
                f.write(f"  {json_str[1:-1]}")  # Remove the outer {}
                if i < len(items) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("}")
        else:
            json.dump(obj, f, ensure_ascii=False)
    print(f"Wrote {path}")

def serialize_inverted(
        inv: dict[str, set]
        ) -> dict[str, list[str]]:
    return {k: sorted(list(v)) for k, v in inv.items()}

def serialize_forward(
        fwd: dict[str, dict]
        ) -> dict[str, dict]:
    out = {}
    for uri, meta in fwd.items():
        out[uri] = {
            "labels": sorted(list(meta.get("labels", []))),
            "norms": sorted(list(meta.get("norms", []))),
        }
    return out

# ------------------------------- MAIN -----------------------------------------

def main():
    # Define paths directly in the code
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset") # ADJUST THIS IN NUVOLOS TO: /space_mounts/atai-hs25/dataset
    embeddings_dir = os.path.join(dataset_dir, "embeddings")
    
    # Input paths
    ent_path = os.path.join(embeddings_dir, "entity_ids.del")
    rel_path = os.path.join(embeddings_dir, "relation_ids.del")
    graph_path = os.path.join(dataset_dir, "graph.nt")
    
    # Output directory
    out_dir = os.path.join(base_dir, "data_cache")
    
    # Language settings
    lang = "en"
    extra_lang = ""

    # Load and normalize IDs to full URIs when possible
    entity_ids = [u for u in read_ids_file(ent_path, "entity") if u]
    relation_ids = [u for u in read_ids_file(rel_path, "relation") if u]

    entities_set = set(entity_ids)
    relations_set = set(relation_ids)

    if not entities_set:
        print("No entities loaded from entity_ids.del; aborting.", file=sys.stderr)
        sys.exit(1)

    langs = tuple(x for x in (lang, extra_lang) if x is not None)

    forward_map, inverted_index, label_rows = collect_labels(
        graph_path=graph_path,
        keep_entities=entities_set,
        keep_relations=relations_set,
        langs=langs,
    )

    # Create entity mappings
    entity_idx_to_id = {idx: eid for idx, eid in enumerate(entity_ids)}
    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
    
    # Create relation mappings
    relation_idx_to_id = {idx: rid for idx, rid in enumerate(relation_ids)}
    relation_id_to_idx = {rid: idx for idx, rid in enumerate(relation_ids)}
    
    # Create label mappings
    entity_label_to_id = {}
    entity_id_to_label = {}
    relation_label_to_id = {}
    relation_id_to_label = {}
    
    # Process forward_map to create label mappings
    for uri, meta in forward_map.items():
        labels = meta['labels']
        if not labels:
            continue
        # Use the first label as the primary label
        primary_label = next(iter(labels))
        
        if uri in entities_set:
            entity_label_to_id[primary_label] = uri
            entity_id_to_label[uri] = primary_label
        elif uri in relations_set:
            relation_label_to_id[primary_label] = uri
            relation_id_to_label[uri] = primary_label

    # Create output directories
    entities_dir = os.path.join(out_dir, "entities")
    relations_dir = os.path.join(out_dir, "relations")
    os.makedirs(entities_dir, exist_ok=True)
    os.makedirs(relations_dir, exist_ok=True)

    # Write entity mappings
    write_json(entity_idx_to_id, os.path.join(entities_dir, "index_to_identifier.json"))
    write_json(entity_id_to_idx, os.path.join(entities_dir, "identifier_to_index.json"))
    write_json(entity_id_to_label, os.path.join(entities_dir, "identifier_to_label.json"))
    write_json(entity_label_to_id, os.path.join(entities_dir, "label_to_identifier.json"))

    # Write relation mappings
    write_json(relation_idx_to_id, os.path.join(relations_dir, "index_to_identifier.json"))
    write_json(relation_id_to_idx, os.path.join(relations_dir, "identifier_to_index.json"))
    write_json(relation_id_to_label, os.path.join(relations_dir, "identifier_to_label.json"))
    write_json(relation_label_to_id, os.path.join(relations_dir, "label_to_identifier.json"))

    # Write CSV files
    write_csv(label_rows, out_dir)

    print("Done.")

if __name__ == "__main__":
    main()