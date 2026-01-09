# Meta-Glasses-Review-Dashboard
An interactive Streamlit dashoboard analyzing 10,000 Meta Glasses Customer Reviews using ML and NLP. Built with Python, scikit-learn, BERTopic, and Streamlit.

# Meta Glasses Review Analytics Dashboard

An interactive Streamlit dashboard that analyzes thousands of Meta Glasses customer reviews using machine learning, NLP, and semantic similarity search. This project combines classical ML, modern topic modeling, and deep‑learning embeddings to provide insights into customer sentiment, review helpfulness, and thematic patterns.

---

## Features

### **1. Helpfulness Prediction Model (≈87% Accuracy)**
- TF‑IDF + Random Forest Regression
- Predicts how helpful a review is likely to be
- Interactive input box for real‑time predictions
- Visualization of top words influencing helpfulness

### **2. Topic Modeling with BERTopic**
- Extracts themes from customer reviews
- Topic hierarchy visualization
- Topic cluster map (2D semantic space)
- Topic similarity heatmap
- Explore top words for any topic

### **3. Review Similarity Search**
- Powered by SentenceTransformer embeddings
- Finds semantically similar reviews
- Filters for:
  - Rating range
  - Exact rating
  - Helpfulness score
  - Topic (optional)
- Returns top matching reviews with similarity scores

### **4. Ratings & Helpfulness Analysis**
- Monthly rating trends
- Rating distribution
- Helpfulness score vs. rating

### **5. Clean Multi‑Tab Streamlit Interface**
- State‑preserving navigation
- Fast caching for models and embeddings
- Responsive, user‑friendly layout

---

## Tech Stack

- **Python**
- **Streamlit**
- **scikit‑learn**
- **BERTopic**
- **Sentence Transformers**
- **pandas / numpy**
- **matplotlib / seaborn / plotly**
- **joblib**

---

## Project Structure
meta_glasses_dashboard/ │ 
  ├── app.py 
  ├── Meta-Glasses-Reviews.csv 
  ├── helpfulness_model.pkl 
  ├── top_words.pkl ├
  |── bertopic_model/ 
  ├── review_similarity.py 
  ├── review_embeddings.pkl   # optional but recommended 
  |── requirements.txt

---

## How It Works

### **1. Load Data**
The dashboard loads and preprocesses Meta Glasses review data, including ratings, dates, and helpfulness scores.

### **2. Machine Learning**
A TF‑IDF + Random Forest model predicts review helpfulness and identifies the most influential words.

### **3. Topic Modeling**
BERTopic extracts themes and generates:
- Hierarchies  
- Clusters  
- Heatmaps  
- Topic‑specific keywords  

### **4. Semantic Search**
SentenceTransformer embeddings enable:
- Cosine similarity search  
- Topic‑aware filtering  
- Fast retrieval of similar reviews  

### **5. Streamlit UI**
A multi‑tab interface organizes the dashboard into:
- Overview  
- Ratings  
- Helpfulness  
- Helpfulness Prediction  
- Topic Modeling  
- Review Similarity Search  

## Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py


