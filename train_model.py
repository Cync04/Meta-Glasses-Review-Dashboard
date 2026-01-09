import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Meta-Glasses-Reviews.csv")
df = df.dropna(subset=['helpfulness_score'])
df['review'] = df['review'].fillna('').astype(str)
y = df['helpfulness_score']
x = df[['review', 'rating']]

text_features = 'review'
numeric_features = ['rating']

preprocess = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), text_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

model = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(n_estimators=300))
])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

import numpy as np

# grab the trained TF-IDF transformer from the pipeline
tfidf = model.named_steps["preprocess"].named_transformers_["text"]

# all the words (features) it learned from the text
feature_names = tfidf.get_feature_names_out()

# feature importances from the random forest
importances = model.named_steps["regressor"].feature_importances_

# split: first part = text features, last part = numeric features (rating)
text_importances = importances[: len(feature_names)]

# get indices of the top 20 most important words
top_idx = np.argsort(text_importances)[-20:]
top_words = feature_names[top_idx]
top_scores = text_importances[top_idx]

for word, score in sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True):
    print(f"{word}: {score:.4f}")

import matplotlib.pyplot as plt
def plot_top_words(top_words, top_scores):
    # Sort the words by importance (highest first)
    sorted_pairs = sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True)

    # Separate them back into two lists
    words_sorted = [w for w, s in sorted_pairs]
    scores_sorted = [s for w, s in sorted_pairs]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(words_sorted, scores_sorted, color='black')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Most Important Words for Predicting Helpfulness")
    ax.invert_yaxis()  # highest importance at the top
    plt.show()
    return fig

import joblib
joblib.dump(model, "helpfulness_model.pkl")
joblib.dump((top_words, top_scores), "top_words.pkl")
fig = plot_top_words(top_words, top_scores)
