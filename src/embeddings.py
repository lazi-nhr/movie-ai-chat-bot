"""Embeddings-based entity linking and prediction."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from . import text_utils
from . import config

@dataclass
class RankedEntity:
    """Entity with a ranking score."""
    uri: str
    score: float

class EmbeddingIndex:
    def __init__(self):
        # Load embeddings
        self.entity_embeddings = np.load(f"{config.EMBEDDINGS['Dir']}/{config.EMBEDDINGS['EntityVec']}")
        self.relation_embeddings = np.load(f"{config.EMBEDDINGS['Dir']}/{config.EMBEDDINGS['RelationVec']}")
        
        # Load and process entity IDs
        with open(f"{config.EMBEDDINGS['Dir']}/{config.EMBEDDINGS['EntityIds']}", 'r') as f:
            self.entity_ids = [line.strip() for line in f if line.strip()]
        
        # Load and process relation IDs    
        with open(f"{config.EMBEDDINGS['Dir']}/{config.EMBEDDINGS['RelationIds']}", 'r') as f:
            self.relation_ids = [line.strip() for line in f if line.strip()]
            
        # Create mappings
        self.entity_to_idx = {uri: idx for idx, uri in enumerate(self.entity_ids)}
        self.relation_to_idx = {uri: idx for idx, uri in enumerate(self.relation_ids)}
        
        # Normalize embeddings
        self.entity_embeddings = self._normalize(self.entity_embeddings)
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
    
    def link_entity(self, mention: str) -> Tuple[str | None, float]:
        """Find best matching entity for a mention."""
        if not mention:
            return None, 0.0
            
        best_uri = None
        best_score = 0.0
        
        for uri in self.entity_ids:
            score = text_utils.get_similarity_score(mention, uri.split('/')[-1])
            if score > best_score:
                best_score = score
                best_uri = uri
                
        return best_uri, best_score
    
    def get_similar_entities(self, entity_uri: str, top_k: int) -> List[RankedEntity]:
        """Find similar entities using embeddings."""
        if entity_uri not in self.entity_to_idx:
            return []
            
        idx = self.entity_to_idx[entity_uri]
        query_embedding = self.entity_embeddings[idx]
        
        # Compute similarities
        similarities = self.entity_embeddings @ query_embedding
        similarities[idx] = -np.inf  # Exclude self
        
        # Get top-k
        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        return [
            RankedEntity(self.entity_ids[i], float(similarities[i]))
            for i in top_indices
        ]
    
    def predict_relation(self, head_uri: str, relation_uri: str, top_k: int) -> List[RankedEntity]:
        """Predict tail entities for a given head and relation."""
        if head_uri not in self.entity_to_idx or relation_uri not in self.relation_to_idx:
            return []
            
        # Get embeddings
        head_idx = self.entity_to_idx[head_uri]
        rel_idx = self.relation_to_idx[relation_uri]
        
        # TransE prediction
        pred = (
            self.entity_embeddings[head_idx] + 
            self.relation_embeddings[rel_idx]
        )
        pred = pred / np.linalg.norm(pred)
        
        # Score all entities
        scores = self.entity_embeddings @ pred
        
        # Get top-k
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        return [
            RankedEntity(self.entity_ids[i], float(scores[i]))
            for i in top_indices
        ]