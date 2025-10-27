import numpy as np
import os
from rdflib import Graph
from sklearn.metrics import pairwise_distances
import json

from extraction import Extraction

# Embeddings for the DDIS Movie Graph
CONFIG = {
    "Hosting": {
        "URL": "https://speakeasy.ifi.uzh.ch",
        "Username": "CyanPeekingMouse",
        "Password": "Qe5Hf3zJ"
    },
    "Data": {
        "Download_URL": "https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/",
        "Directory": "dataset/",
        "Cache": "cache/",
    }
}

class Embeddings():
    def __init__(self):
        self.graph = self.load_graph()
        (self.entity_emb, self.relation_emb) = self.load_embeddings()
        (self.ent2id, self.id2ent,
         self.ent2lbl, self.lbl2ent,
         self.rel2id, self.id2rel,
         self.rel2lbl, self.lbl2rel) = self.load_mappings()

    def load_graph(self):
        graph = Graph()
        path = CONFIG["Data"]["Directory"] + "graph.nt"
        graph.parse(path, format="nt")
        return graph
    
    def load_embeddings(self):
        data_dir = CONFIG["Data"]["Directory"]
        entity_emb = np.load(os.path.join(data_dir, 'embeddings', 'entity_embeds.npy'))
        relation_emb = np.load(os.path.join(data_dir, 'embeddings', 'relation_embeds.npy'))
        return entity_emb, relation_emb
    
    def load_mappings(self):
        # id: index
        # ent: entity identifier
        # rel: relation identifier
        # lbl: label
        cache_dir = CONFIG["Data"]["Cache"]
        with open(os.path.join(cache_dir, 'entities', 'identifier_to_index.json'), "r") as f:
            ent2id = json.load(f)
            id2ent = {v: k for k, v in ent2id.items()}

        with open(os.path.join(cache_dir, 'entities', 'identifier_to_label.json'), "r") as f:
            ent2lbl = json.load(f)
            lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

        with open(os.path.join(cache_dir, 'relations', 'identifier_to_index.json'), "r") as f:
            rel2id = json.load(f)
            id2rel = {v: k for k, v in rel2id.items()}

        with open(os.path.join(cache_dir, 'relations', 'identifier_to_label.json'), "r") as f:
            rel2lbl = json.load(f)
            lbl2rel = {lbl: ent for ent, lbl in rel2lbl.items()}
        
        return ent2id, id2ent, ent2lbl, lbl2ent, rel2id, id2rel, rel2lbl, lbl2rel
    
    def get_best_tail(self, head_uri: str, relation_uri: str, top_k: int = 5) -> list[tuple]:
        """
        Find the most likely tail entities given a head entity and relation.
        
        Args:
            head_uri: URI of the head entity
            relation_uri: URI of the relation
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (entity_uri, label, score, rank)
        """
        # Get embeddings for head and relation
        head = self.entity_emb[self.ent2id[head_uri]]
        rel = self.relation_emb[self.rel2id[relation_uri]]
        
        # TransE scoring function: head + rel should be close to tail
        lhs = head + rel
        
        # Compute distances to all possible tail entities
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        
        # Get top-k predictions
        most_likely = dist.argsort()[:top_k]
        
        results = []
        for rank, idx in enumerate(most_likely):
            entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(entity_uri, "Unknown")
            score = dist[idx]
            results.append((entity_uri, label, score, rank + 1))
        
        return results
    
    def get_best_head(self, tail_uri: str, relation_uri: str, top_k: int = 5) -> list[tuple]:
        """
        Find the most likely head entities given a tail entity and relation.
        
        Args:
            tail_uri: URI of the tail entity
            relation_uri: URI of the relation
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (entity_uri, label, score, rank)
        """
        # Get embeddings for tail and relation
        tail = self.entity_emb[self.ent2id[tail_uri]]
        rel = self.relation_emb[self.rel2id[relation_uri]]
        
        # TransE scoring function: head should be close to tail - rel
        # Since h + r = t, then h = t - r
        rhs = tail - rel
        
        # Compute distances to all possible head entities
        dist = pairwise_distances(rhs.reshape(1, -1), self.entity_emb).reshape(-1)
        
        # Get top-k predictions
        most_likely = dist.argsort()[:top_k]
        
        results = []
        for rank, idx in enumerate(most_likely):
            entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(entity_uri, "Unknown")
            score = dist[idx]
            results.append((entity_uri, label, score, rank + 1))
        
        return results
    




## Example Questions
extraction = Extraction()
embeddings = Embeddings()

questions = {
    "Who is the director of Good Will Hunting?": "Gus Van Sant is the director of Good Will Hunting.",
    "Who directed The Bridge on the River Kwai?": "David Lean directed The Bridge on the River Kwai.",
    "Who is the director of Star Wars: Episode VI - Return of the Jedi": "David Lean directed The Bridge on the River Kwai.",
    "Who is the screenwriter of The Masked Gang: Cyprus?": "The answer suggested by embeddings: Cengiz Küçükayvaz, Murat Aslan, and Melih Ekener.",
    "What is the MPAA film rating of Weathering with You?": "According to embeddings, the MPAA film rating of Weathering with You is PG-13.",
    "What is the genre of Good Neighbors?": "The genre of Good Neighbors is likely to be drama, comedy-drama, and comedy film.",
    "Who is the director of Batman 1989?": "Hi, the director of Batman 1989 is Timothy Walter Burton",
}
for question in questions.keys():
    extracted_entity = extraction.extract_entity(question)
    entity_label, entity_URI, score, distance = extraction.link_entity(extracted_entity)

    extracted_relation = extraction.extract_relation(question)
    relation_label, relation_URI, rel_score, rel_distance = extraction.link_relation(extracted_relation)

    results = embeddings.get_best_head(entity_URI, relation_URI, top_k=1)

    for result in results:
        head_uri, head_label, head_score, head_rank = result
        print(f"Q: {question}")
        print(f"A: {head_label} ({head_uri})")