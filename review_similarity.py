import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for all reviews
def compute_embeddings(embedder, reviews):
    return embedder.encode(reviews, show_progress_bar=False)

# Keyword search in reviews
def extract_keywords(text):
    if not isinstance(text, str):
        return set()
    stopwords = {'this', 'that', 'with', 'from', 'have', 'were', 'they', 'your', 'just', 'like', 'when', 'what', 'about', 'there', 'their', 'would', 'could', 'should'}
    words = text.lower().split()
    return set([w for w in words if len(w) > 3 and w not in stopwords])

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
    df_unique = df.drop_duplicates(subset=["review"]).reset_index(drop=True).fillna("").astype({"review": str})
    review_embeddings = review_embeddings[df_unique.index]

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
    temp = df_unique.copy()
    temp["similarity"] = similarities

    # Perfrom keyword filtering
    query_keywords = extract_keywords(query)
    temp['keyword_score'] = temp['review'].apply(
        lambda x: len(query_keywords.intersection(extract_keywords(x)))
    )

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
    
    # Normalize keyword score
    temp['keyword_score'] = temp['keyword_score'] / max(temp['keyword_score'].max(), 1)
     
    # Combine similarity and keyword score
    temp['final_score'] = temp['similarity'] + 0.05 * temp['keyword_score']

    # Sort by similarity
    temp = temp.sort_values(by="final_score", ascending=False)

    return temp.head(top_k)
