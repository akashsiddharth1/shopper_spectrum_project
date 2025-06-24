# shopper_spectrum_project

🧠 Customer Segmentation & 🎯 Product Recommendation System
This project combines Unsupervised Machine Learning (Clustering) and Collaborative Filtering (Recommendation System) to analyze retail customer behavior and provide personalized product suggestions. The final solution is deployed as a user-friendly Streamlit web app.

🔍 Problem Statement
Retailers often struggle to understand their customers and personalize their marketing strategies. This project solves that by:

Segmenting customers based on RFM (Recency, Frequency, Monetary) analysis using clustering.

Recommending similar products based on purchase history using item-based collaborative filtering.

📁 Dataset
Online Retail Dataset (transactions from a UK-based e-commerce store)

Fields include: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

🛠️ Key Features
1️⃣ Customer Segmentation (RFM Clustering)
Feature engineering:

Recency: Days since last purchase

Frequency: Total number of purchases

Monetary: Total amount spent

Normalized RFM values

Clustering using KMeans, optimized with Elbow Method and Silhouette Score

Labeled clusters:

High-Value, Regular, Occasional, At-Risk

Visualized with 2D & 3D scatter plots

2️⃣ Product Recommendation System
Built using item-based collaborative filtering

Customer–product matrix created from purchase data

Similarity computed using cosine similarity

For any selected product, returns Top 5 most similar products

🚀 Streamlit Web App
🎯 Product Recommendation Module
Select a product → Get 5 most similar products

🧠 Customer Segmentation Module
Input: Recency, Frequency, Monetary

Output: Predicted cluster + segment label

📦 Tech Stack
Python, Pandas, NumPy

Scikit-learn – KMeans, cosine similarity

Matplotlib / Seaborn – EDA & visualization

Streamlit – Web app deployment

✅ Deliverables
Cleaned and processed dataset

Trained clustering model (best_rfm_kmeans_model.pkl)

Trained scaler (rfm_scaler.pkl)

Reusable similarity matrix

Interactive app for business stakeholders

📊 Sample Use Cases
Targeted marketing based on customer segment

Bundle recommendations or upselling related items

Churn risk detection (via At-Risk cluster)
