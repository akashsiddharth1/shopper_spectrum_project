import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --------------------------------------------
# PAGE SETUP
# --------------------------------------------
st.set_page_config(page_title="RFM Segmentation & Recommendation", layout="wide")
st.title("🧠 Customer Segmentation & 🎯 Product Recommendation App")

# --------------------------------------------
# LOAD DATA & MODELS
# --------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

@st.cache_resource
def load_models():
    with open("best_rfm_kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("rfm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

df = load_data()
kmeans_model, rfm_scaler = load_models()

# --------------------------------------------
# PRODUCT MAPPINGS & SIMILARITY MATRIX
# --------------------------------------------

@st.cache_data
def build_similarity_matrix(df):
    matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
    matrix.fillna(0, inplace=True)
    similarity = cosine_similarity(matrix.T)
    return pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns), matrix.columns

item_similarity_df, stock_codes = build_similarity_matrix(df)

product_map = df.drop_duplicates(subset='StockCode')[['StockCode', 'Description']].dropna()
stock_desc_dict = pd.Series(product_map.Description.values, index=product_map.StockCode).to_dict()
desc_stock_dict = {v: k for k, v in stock_desc_dict.items()}

# --------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------

def recommend_products_by_name(product_name, top_n=5):
    if product_name not in desc_stock_dict:
        return "❌ Product name not found!"
    
    stock_code = desc_stock_dict[product_name]
    if stock_code not in item_similarity_df.columns:
        return "❌ Product code not in similarity matrix!"
    
    similar_items = item_similarity_df[stock_code].sort_values(ascending=False)[1:top_n+1]
    results = []
    for code in similar_items.index:
        name = stock_desc_dict.get(code, f"StockCode {code}")
        score = similar_items[code]
        results.append((name, score))
    return results

# --------------------------------------------
# SEGMENT LABEL MAPPING
# --------------------------------------------

def assign_segment_label(recency, frequency, monetary):
    if recency < 30 and frequency > 10 and monetary > 1000:
        return 'High-Value'
    elif frequency >= 5 and monetary >= 300:
        return 'Regular'
    elif frequency <= 2 and monetary <= 200 and recency > 90:
        return 'Occasional'
    elif recency > 120 and frequency < 3 and monetary < 200:
        return 'At-Risk'
    else:
        return 'Other'

# --------------------------------------------
# STREAMLIT UI – TABS
# --------------------------------------------

tab1, tab2 = st.tabs(["🎯 Product Recommendation", "🧠 Customer Segmentation"])

# ----------------------
# 🎯 Tab 1: Recommendation
# ----------------------
with tab1:
    st.header("🎯 Recommend Similar Products")

    selected_product = st.selectbox("Choose a product:", sorted(desc_stock_dict.keys()))

    if st.button("Get Recommendations"):
        results = recommend_products_by_name(selected_product)
        if isinstance(results, str):
            st.error(results)
        else:
            st.subheader("🔁 Top 5 Similar Products:")
            for i, (name, score) in enumerate(results, 1):
                st.markdown(f"**{i}. {name}**  — Similarity Score: `{score:.2f}`")

# ----------------------
# 🧠 Tab 2: Segmentation
# ----------------------
with tab2:
    st.header("🧠 Predict Customer Segment")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=90)
    with col2:
        frequency = st.number_input("Frequency (purchases)", min_value=1, max_value=100, value=5)
    with col3:
        monetary = st.number_input("Monetary (£ spent)", min_value=1.0, max_value=10000.0, value=300.0)

    if st.button("Predict Segment"):
        user_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        scaled_input = rfm_scaler.transform(user_input)
        cluster = kmeans_model.predict(scaled_input)[0]
        label = assign_segment_label(recency, frequency, monetary)

        st.success(f"📌 Predicted Segment: **{label}** (Cluster ID: {cluster})")
