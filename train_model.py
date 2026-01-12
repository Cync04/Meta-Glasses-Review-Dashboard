import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Meta-Glasses-Reviews.csv")

# Clean missing helpfulness
df = df.dropna(subset=['helpfulness_score'])

# Ensure review is string
df['review'] = df['review'].fillna('').astype(str)

# ⭐ NEW: Add review length feature
df['review_length'] = df['review'].apply(lambda x: len(x.split()))

y = df['helpfulness_score']
X = df[['review', 'rating', 'review_length']]

# Feature groups
text_features = 'review'
numeric_features = ['rating', 'review_length']   # ⭐ review_length added here

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), text_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Model pipeline
model = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(n_estimators=300))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Extract TF-IDF + feature importances
import numpy as np

tfidf = model.named_steps["preprocess"].named_transformers_["text"]
feature_names = tfidf.get_feature_names_out()

importances = model.named_steps["regressor"].feature_importances_

# Text importances only
text_importances = importances[: len(feature_names)]

top_idx = np.argsort(text_importances)[-20:]
top_words = feature_names[top_idx]
top_scores = text_importances[top_idx]

for word, score in sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True):
    print(f"{word}: {score:.4f}")

# Plot
import matplotlib.pyplot as plt
def plot_top_words(top_words, top_scores):
    sorted_pairs = sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True)
    words_sorted = [w for w, s in sorted_pairs]
    scores_sorted = [s for w, s in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(words_sorted, scores_sorted, color='black')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Most Important Words for Predicting Helpfulness")
    ax.invert_yaxis()
    plt.show()
    return fig

import joblib
joblib.dump(model, "helpfulness_model.pkl")
joblib.dump((top_words, top_scores), "top_words.pkl")

fig = plot_top_words(top_words, top_scores)