import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("Meta-Glasses-Reviews.csv")
df['review'] = df['review'].fillna('').astype(str)

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['review'])

#deploying the model
lda = LatentDirichletAllocation(
    n_components=6,
    random_state=42,
    learning_method='batch'
)

lda.fit(X)

def display_topics(model, feature_names, num_words=10):
    for idx, topic in enumerate(model.components_):
        print(f'\nTopic {idx+1}:')
        top_features= topic.argsort()[-num_words:][::-1]
        print([feature_names[i] for i in top_features])

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names)