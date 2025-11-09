import numpy as np
import os
from rdflib import Graph
from sklearn.metrics import pairwise_distances
import json
from config import CONFIG


class Embeddings():
    def __init__(self):
        # self.graph = self.load_graph()
        (self.entity_emb, self.relation_emb) = self.load_embeddings()
        (self.ent2id, self.id2ent,
         self.ent2lbl, self.lbl2ent,
         self.rel2id, self.id2rel,
         self.rel2lbl, self.lbl2rel) = self.load_mappings()

    def load_graph(self):
        graph = Graph()
        path = CONFIG["Data"]["Directory"] + "graph.nt"
        print(f"Loading graph from {path}...")
        graph.parse(path, format="nt")
        print("Graph loaded.")
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
    
    def get_best_tail(
            self, 
            head_uri: str, 
            relation_uri: str, 
            top_k: int = 1
            ) -> list[tuple]:
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
        
        # Get sorted indices and filter those that are in our mapping
        sorted_indices = dist.argsort()
        valid_indices = [idx for idx in sorted_indices if idx in self.id2ent]
        most_likely = valid_indices[:top_k]
        
        results = []
        for rank, idx in enumerate(most_likely):
            entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(entity_uri, "Unknown")
            score = dist[idx]
            results.append((entity_uri, label, score, rank + 1))
        
        return results
    
    def get_best_head(
            self, 
            tail_uri: str, 
            relation_uri: str, 
            top_k: int = 1
            ) -> list[tuple]:
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
        
        # Get sorted indices and filter those that are in our mapping
        sorted_indices = dist.argsort()
        valid_indices = [idx for idx in sorted_indices if idx in self.id2ent]
        most_likely = valid_indices[:top_k]
        
        results = []
        for rank, idx in enumerate(most_likely):
            entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(entity_uri, "Unknown")
            score = dist[idx]
            results.append((entity_uri, label, score, rank + 1))
        
        return results
    
    def get_best_result(
            self, 
            entity_uri: str, 
            relation_uri: str, 
            top_k: int = 1
            ) -> list[tuple]:
        """
        Try both head and tail predictions and return the results with the better score.
        
        Args:
            entity_uri: URI of the known entity (head or tail)
            relation_uri: URI of the relation
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (entity_uri, label, score, rank)
        """
        if entity_uri not in self.ent2id:
            raise ValueError(f"Unknown entity URI: {entity_uri}")

        # Try both predictions
        tail_results = self.get_best_tail(entity_uri, relation_uri, top_k)
        head_results = self.get_best_head(entity_uri, relation_uri, top_k)

        # Compare best scores (lower is better since we use distances)
        best_tail_score = tail_results[0][2] if tail_results else float('inf')
        best_head_score = head_results[0][2] if head_results else float('inf')

        print(f"Best tail: {tail_results[0][1]}, Best head: {head_results[0][1]}")

        return tail_results if best_tail_score < best_head_score else head_results