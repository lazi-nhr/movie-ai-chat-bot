# Movie Chatbot

A specialized conversational AI agent designed to answer movie-related questions using advanced AI/ML concepts including embeddings, recommender systems, and Named Entity Recognition (NER). The chatbot runs on **Speakeasy**, an external client that provides an intuitive user interface for seamless interaction.

INSERT GIF of interaction with the chatbot here

## System Architecture

The chatbot employs a multi-stage pipeline to process and respond to user queries:

<p align="center">
  <img src="media/flowchart.png" alt="System Flowchart" width="90%">
</p>

1. **Classification**: An initial classifier analyzes incoming requests and assigns them to the appropriate request type
2. **Entity & Relation Extraction**: A Named Entity Recognition (NER) system extracts relevant entities such as movie names, actor names, places, and dates when necessary. Additionally, relation extraction identifies relationships between entities (e.g., "director of [movie]", "publication date of [movie]").
3. **Request Handling**: Based on the classified request type, the system routes the query to the appropriate module. The specific requsts type handled is presented in the next section.
4. **Answer Generation**: With the necessary information extracted, the system generates a response.

## Supported Query Types

### 1. Factual Questions

The system leverages a **knowledge graph** to answer structured factual questions about movies. User queries are translated into **SPARQL queries** that retrieve precise information such as:
- Directors and crew members
- Release dates
- Genres and categories
- Cast and actors
- Production details

The knowledge graph enables fast and accurate retrieval of structured data, making it ideal for straightforward factual queries.

<p align="center">
  <img src="media/factual.png" alt="Factual Questions" width="50%">
</p>

### 2. Embeddings-Based Search

For factual questions that cannot be resolved through the knowledge graph, the system employs **knowledge graph embeddings** as a fallback mechanism.

Embeddings are created from the knowledge graph itself, representing entities and relations as dense vectors in a continuous space. This vector-based approach enables powerful semantic operations:
- **Vector Arithmetic**: Entity vectors can be combined with relation vectors through simple addition and subtraction (e.g., "Movie + directed_by = Director")
- **Semantic Similarity**: Measuring distances between vectors to find related entities
- **Relational Reasoning**: Navigating the knowledge graph through vector operations rather than explicit queries

This approach bridges the gap between structured data retrieval and natural language understanding, allowing the system to answer complex questions through learned vector representations.

<p align="center">
  <img src="media/embedding.png" alt="Embeddings" width="50%">
</p>

### 3. Movie Recommendations

The chatbot features a **recommender system** powered by **Singular Value Decomposition (SVD)** that suggests movies based on user preferences. Users can:
- Ask for recommendations similar to movies they've enjoyed
- Discover films based on genre, mood, or themes
- Receive personalized suggestions based on latent features extracted through matrix factorization

SVD decomposes the user-movie interaction matrix to identify underlying patterns and similarities, enabling accurate movie recommendations even with sparse data.

<p align="center">
  <img src="media/recommendation.png" alt="Recommendations" width="50%">
</p>

### 4. Multimedia Content

The system provides **visual content** to enhance the user experience, including:
- Movie posters and promotional images
- Actor photographs
- Scene stills and screenshots

<p align="center">
  <img src="media/multimedia.png" alt="Multimedia" width="50%">
</p>

## Technologies Used

- **Natural Language Processing**: Entity extraction and text classification
- **Knowledge Graphs**: Structured data storage and retrieval
- **Semantic Embeddings**: Vector-based similarity search
- **Recommender Systems**: Collaborative and content-based filtering
- **Speakeasy**: External UI client for user interactions

## Getting Started

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python src/agent.py
```

---

*This project demonstrates the integration of multiple AI/ML techniques to create an intelligent, domain-specific conversational agent.*
