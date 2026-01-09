import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for all reviews
def compute_embeddings(embedder, reviews):
    return embedder.encode(reviews, show_progress_bar=False)

# Perform similarity search with filters
def find_similar_reviews(
    query,
    embedder,
    review_embeddings,
    df,
    top_k=5,
    min_rating=None,
    max_rating=None,
    exact_rating=None,
    min_helpfulness=None,
    topic_filter=None,
    topic_model=None
):
    # Embed the query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Force the reviews embeddings to be 2D
    review_embeddings = np.array(review_embeddings)
    if review_embeddings.ndim == 1:
        review_embeddings = review_embeddings.reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, review_embeddings)[0]

    # Build a working DataFrame
    temp = df.copy()
    temp["similarity"] = similarities

    # Apply filters
    if min_rating is not None:
        temp = temp[temp["rating"] >= min_rating]

    if max_rating is not None:
        temp = temp[temp["rating"] <= max_rating]

    if exact_rating is not None:
        temp = temp[temp["rating"] == exact_rating]

    if min_helpfulness is not None:
        temp = temp[temp["helpfulness_score"] >= min_helpfulness]

    if topic_filter is not None and topic_model is not None:
        topics = topic_model.transform(temp["review"].tolist())
        temp["topic"] = topics
        temp = temp[temp["topic"] == topic_filter]

    # Sort by similarity
    temp = temp.sort_values(by="similarity", ascending=False)

    return temp.head(top_k)