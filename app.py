import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from bertopic import BERTopic

# ---------------------------------------------------
# CACHING: Load model + top words only once
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("helpfulness_model.pkl")

@st.cache_resource
def load_top_words():
    return joblib.load("top_words.pkl")

@st.cache_resource
def load_bertopic():
    return BERTopic.load("bertopic_model_minimal.pkl")

model = load_model()
top_words, top_scores = load_top_words()
topic_model = load_bertopic()
topics_info = topic_model.get_topic_info()

# ---------------------------------------------------
# SESSION STATE FOR TAB PERSISTENCE
# ---------------------------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"

# ---------------------------------------------------
# PLOT FUNCTION
# ---------------------------------------------------
def plot_top_words(top_words, top_scores):
    sorted_pairs = sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True)
    words_sorted = [w for w, s in sorted_pairs]
    scores_sorted = [s for w, s in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words_sorted, scores_sorted, color="black")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Most Important Words for Predicting Helpfulness")
    ax.invert_yaxis()
    return fig

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("Meta-Glasses-Reviews.csv")
df["date"] = pd.to_datetime(df["date"])

df_monthly = df.groupby(df["date"].dt.to_period("M"))["rating"].mean().reset_index()
df_monthly["date"] = df_monthly["date"].dt.to_timestamp()

# ---------------------------------------------------
# CUSTOM NAVIGATION BAR (STATE-PRESERVING)
# ---------------------------------------------------
st.title("Meta Glasses Reviews Dashboard")

tabs = ["Overview", "Ratings", "Helpfulness", "Helpfulness Prediction Model", "Topic Modeling", 'Review Similarity Search']

selected_tab = st.radio(
    "Navigation",
    tabs,
    index=tabs.index(st.session_state.active_tab),
    horizontal=True
)

st.session_state.active_tab = selected_tab

# ---------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------
if selected_tab == "Overview":
    st.header("📊 Overview Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Average Rating", round(df["rating"].mean(), 2))
    col3.metric("Positive Review %", f"{round(df['is_positive_review'].mean()*100, 1)}%")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------------------------------------------
# TAB 2 — RATINGS
# ---------------------------------------------------
elif selected_tab == "Ratings":
    st.header("📈 Rating Trend Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_monthly, x="date", y="rating", marker="o", ax=ax)
    ax.set_title("Average Rating by Month")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Rating")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")
    st.header("📊 Rating Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="rating", ax=ax2)
    ax2.set_title("Distribution of Ratings")
    st.pyplot(fig2)

# ---------------------------------------------------
# TAB 3 — HELPFULNESS
# ---------------------------------------------------
elif selected_tab == "Helpfulness":
    st.header("💬 Helpfulness Score vs Rating")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="rating", y="helpfulness_score", ax=ax3)
    ax3.set_title("Average Helpfulness Score by Rating")
    st.pyplot(fig3)

# ---------------------------------------------------
# TAB 4 — Helpfulness Predicition Model
# ---------------------------------------------------
elif selected_tab == "Helpfulness Prediction Model":
    st.header("🤖 Helpfulness Prediction Model")
    st.write("""
    This model predicts how helpful a review is likely to be based on:
    - The review text  
    - The rating  
    It was trained using TF‑IDF + Random Forest Regression.
    """)

    st.subheader("🔍 Top Words That Predict Helpfulness")
    fig = plot_top_words(top_words, top_scores)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧠 Predict Helpfulness of a New Review")

    user_review = st.text_area("Write a review:")
    user_rating = st.slider("Rating", 1, 5, 5)

    if st.button("Predict Helpfulness"):
        st.session_state.active_tab = "Helpfulness Prediction Model"
        if len(user_review.strip()) == 0:
            st.warning("Please enter a review before predicting.")
        else:
            new_df = pd.DataFrame({"review": [user_review], "rating": [user_rating]})
            prediction = model.predict(new_df)[0]
            st.success(f"Predicted Helpfulness Score: {prediction:.2f}")

# ---------------------------------------------------
# TAB 5 — TOPIC MODELING
# ---------------------------------------------------
elif selected_tab == "Topic Modeling":
    st.header("🗂️ Topic Modeling with BERTopic")
    st.write("""
    This section uses BERTopic to identify common themes in customer reviews.
    Explore the hierarchy, clusters, heatmap, and individual topic details.
    """)

    topics_info = topic_model.get_topic_info()
    st.subheader("Identified Topics")
    st.dataframe(topics_info)

    st.markdown("---")

    # Topic Hierarchy
    st.subheader("🌳 Topic Hierarchy")
    st.caption("This visualization shows how topics group into broader parent themes.")
    try:
        with st.spinner("Generating topic hierarchy..."):
            fig_hierarchy = topic_model.visualize_hierarchy()
            st.plotly_chart(fig_hierarchy, use_container_width=True)
    except Exception:
        st.warning("Hierarchy visualization is not available for this model.")

    st.markdown("---")

    # Topic Clusters
    st.subheader("📌 Topic Clusters")
    st.caption("This 2D map shows how topics relate to each other based on semantic similarity.")
    try:
        with st.spinner("Generating topic clusters..."):
            fig_clusters = topic_model.visualize_topics()
            st.plotly_chart(fig_clusters, use_container_width=True)
    except Exception:
        st.warning("Cluster visualization is not available for this model.")

    st.markdown("---")

    # Topic Heatmap
    st.subheader("🌡️ Topic Similarity Heatmap")
    st.caption("This heatmap shows how similar topics are to each other.")
    try:
        with st.spinner("Generating heatmap..."):
            fig_heatmap = topic_model.visualize_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception:
        st.warning("Heatmap visualization is not available for this model.")

    st.markdown("---")

    # Explore Specific Topic
    st.subheader("🔍 Explore a Specific Topic")
    st.caption("Select a topic to view its top words and their importance weights.")

    topic_list = topics_info["Topic"].tolist()
    topic_number = st.selectbox("Select a Topic:", topic_list)

    if st.button("Show Topic Details"):
        st.session_state.active_tab = "Topic Modeling"
        topic_details = topic_model.get_topic(topic_number)
        if topic_details:
            st.write(f"### Topic {topic_number} Details")
            for word, weight in topic_details:
                if word != "_":
                    st.write(f"- **{word}**: {weight:.4f}")
        else:
            st.warning("Topic not found.")
# ---------------------------------------------------
# Tab 6 - REVIEW SIMILARITY SEARCH WITH FILTERS
# ---------------------------------------------------
elif selected_tab == "Review Similarity Search":
    st.markdown("---")
    st.header("🔎 Review Similarity Search")
    st.caption("Find reviews similar in meaning, with optional filters.")

    import sys
    sys.path.append(r"w:\Programming\Python\Data_Analysis\Meta_Glasses")

    from review_similarity import (
        load_embedder,
        compute_embeddings,
        find_similar_reviews
    )

    # Load embedder + embeddings
    embedder = load_embedder()

    @st.cache_resource
    def get_embeddings():
        return compute_embeddings(embedder, df["review"].astype(str).tolist())

    review_embeddings = get_embeddings()

    # User input
    user_query = st.text_area("Enter a review to search for similar ones:")

    # Filters
    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_rating = st.slider("Minimum Rating", 1, 5, 1)
        max_rating = st.slider("Maximum Rating", 1, 5, 5)

    with col2:
        exact_rating = st.selectbox("Exact Rating (optional)", [None, 1, 2, 3, 4, 5])

    with col3:
        min_helpfulness = st.slider("Minimum Helpfulness Score", 0, 10, 0)

    topics_info = topic_model.get_topic_info()

    topic_filter = st.selectbox(
        "Filter by Topic (optional)",
        [None] + topics_info["Topic"].tolist()
    )

    # Run search
    if st.button("Find Similar Reviews"):
        st.session_state.active_tab = "Topic Modeling"

        if len(user_query.strip()) == 0:
            st.warning("Please enter a review.")
        else:
            results = find_similar_reviews(
                query=user_query,
                embedder=embedder,
                review_embeddings=review_embeddings,
                df=df,
                top_k=5,
                min_rating=min_rating,
                max_rating=max_rating,
                exact_rating=exact_rating,
                min_helpfulness=min_helpfulness,
                topic_filter=topic_filter,
                topic_model=topic_model
            )

            st.subheader("Top Matching Reviews")

            if len(results) == 0:
                st.info("No reviews matched your filters.")
            else:
                for _, row in results.iterrows():
                    st.write(f"**Similarity:** {row['similarity']:.3f}")
                    st.write(f"**Rating:** {row['rating']}")
                    st.write(f"**Helpfulness:** {row['helpfulness_score']}")
                    st.write(row["review"])

                    st.markdown("---")
