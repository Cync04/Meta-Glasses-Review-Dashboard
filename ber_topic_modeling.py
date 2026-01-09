import pandas as pd
from bertopic import BERTopic

df = pd.read_csv('Meta-Glasses-Reviews.csv')
df['review'] = df['review'].fillna('').astype(str)

docs = df['review'].tolist()

topic_model = BERTopic(language='english', verbose=True)
topics, probs = topic_model.fit_transform(docs)

topic_model.get_topic_info(0)
topic_model.save('bertopic_model')

