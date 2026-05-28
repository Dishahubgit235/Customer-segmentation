import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Customer Segmentation Web App")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write(data.head())

    #  ADD HERE
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    #  USE HERE
    col1 = st.selectbox("Select X-axis feature", numeric_cols)
    col2 = st.selectbox("Select Y-axis feature", numeric_cols)

    X = data[[col1, col2]]

    # Select number of clusters
    k = st.slider("Select number of clusters", 2, 10, 5)

    # Apply KMeansst
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    # Plot graph
    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
    ax.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1], s=200)

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("Customer Segmentation")

    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")