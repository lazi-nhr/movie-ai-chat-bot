import re
from rdflib import Graph
import json
from embeddings import Embeddings

# Configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

class Factual():
    def __init__(self):
        self.graph = self.load_graph()
        self.embeddings = Embeddings()

    def load_graph(self):
        graph = Graph()
        path = CONFIG["Data"]["Directory"] + "graph.nt"
        print(f"Loading graph from {path}...")
        graph.parse(path, format="nt")
        print("Graph loaded.")
        return graph
    
    def sparql_query(self, query: str):
        results = self.graph.query(query)
        for row in results:
            print(row)
        return results
    
    def translate_to_sparql(self, entity_uri: str, relation_uri: str) -> str:
        # Implement here maybe use method "format_results" below?
        # Keep in mind that we might have to handle pure SPARQL queries too
        qry = None
        if entity_uri and relation_uri:
            qry = f"""
            SELECT ?object ?label
            WHERE {{
            <{entity_uri}> <{relation_uri}> ?object .
            OPTIONAL {{
                ?object rdfs:label ?label .
                FILTER(LANG(?label) = 'en')
            }}
            }}
            """
            print(f"1. entity: {entity_uri}, relation: {relation_uri}, query {qry}")
        if not qry:
            raise ValueError(f"entity: {entity_uri}, relation: {relation_uri}")
        return qry
    
    # This method is not used in the current code. 
    # It is copied from the first evaluation event. 
    # Ajdust it to return a list of strings with the results (labels).
    def format_results(self, results) -> str:
        try:
            if results.type == 'ASK':
                return "true" if bool(results) else "false"

            if results.type == 'SELECT':
                vars_ = [str(v) for v in results.vars]
                lines = []

                for i, row in enumerate(results):
                    if i >= 50:
                        lines.append("... (truncated)")
                        print(row)
                        break

                    if hasattr(row, "asdict"):
                        b = {str(k): v for k, v in row.asdict().items()}
                    else:
                        b = {vars_[idx]: row[idx] for idx in range(min(len(vars_), len(row)))}

                    qid = None
                    for val in b.values():
                        m = re.search(r"/entity/(Q\d+)$", str(val))
                        if m:
                            qid = m.group(1)
                            break

                    label = None
                    preferred_keys = ["directorLabel", "label", "name"] + \
                                    [k for k in b.keys() if k.lower().endswith("label")]
                    for k in preferred_keys:
                        if k in b:
                            label = str(b[k])
                            break

                    # this was added to retrieve the label from the already existing ent2lbl
                    if not label and qid:
                        uri = f"http://www.wikidata.org/entity/{qid}"
                        label = self.embeddings.ent2lbl.get(uri, None)

                    if qid and label:
                        lines.append(f"{label} ({qid})")
                    elif label:
                        lines.append(label)
                    elif qid:
                        lines.append(f"No label found {qid}")
                    else:
                        pieces = [str(b[v]) for v in vars_ if v in b]
                        lines.append(" | ".join(pieces) if pieces else str(b))

                return " and ".join(lines) if lines else "No results found."

            if results.type in ('CONSTRUCT', 'DESCRIBE'):
                g = Graph()
                for t in results:
                    g.add(t)
                preview = []
                for i, (s, p, o) in enumerate(g):
                    if i >= 10:
                        preview.append("... (truncated)")
                        break
                    preview.append(f"{s} {p} {o}")
                return f"Triples returned: {len(g)}\n" + "\n".join(preview)

            return "No results found."
        except Exception as e:
            return f"Error formatting results: {e}"