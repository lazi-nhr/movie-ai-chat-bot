"""Configuration settings for the chat bot."""

HOSTING = {
    "URL": "https://speakeasy.ifi.uzh.ch",
    "Username": "CyanPeekingMouse",
    "Password": "Qe5Hf3zJ",
}

EMBEDDINGS = {
    "Dir": "/space_mounts/atai-hs25/dataset/embeddings",
    "EntityVec": "entity_embeds.npy",
    "EntityIds": "entity_ids.del",
    "RelationVec": "relation_embeds.npy",
    "RelationIds": "relation_ids.del",
}

THRESHOLDS = {
    "EntityLinkMin": 0.30,   # minimum fuzzy score for entity linking
    "RelationLinkMin": 0.40,  # minimum fuzzy score for relation linking
    "TopScoreMin": 0.30,     # minimum top score to accept an embedding answer
}

ANSWER = {
    "TopKSimilar": 5,
    "TopKPrediction": 3,
}

KNOWLEDGE_GRAPH_PATH = "/space_mounts/atai-hs25/dataset/graph.nt"