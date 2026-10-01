URL : "movie-recommendation-system-zh4znvd4k5lwnbh5x3hiv7.streamlit.app"
# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using **Python, Machine Learning, and Streamlit**.

The system recommends movies similar to a movie selected by the user. It analyzes information such as the movie overview, genres, keywords, cast, and director to identify movies with similar content.

---

## 📌 Project Overview

This project uses a **Content-Based Filtering** approach.

The movie information is combined into a single text feature called **tags**. The tags are converted into numerical vectors using **CountVectorizer**, and **Cosine Similarity** is used to measure the similarity between movies.

The application recommends the **Top 5 most similar movies** for the selected movie.

---

## 🚀 Features

- 🎬 Select a movie from the movie dataset
- 🔎 Find similar movies automatically
- 🤖 Content-based recommendation using Machine Learning
- 📊 Text feature engineering
- 🔢 CountVectorizer for text vectorization
- 📐 Cosine Similarity for recommendation
- 🎨 Interactive Streamlit user interface
- ⭐ Displays similarity percentage for recommendations

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Git & GitHub**

---

## 📂 Dataset

This project uses the **TMDB 5000 Movie Dataset**.

The project uses two CSV files:
tmdb_5000_movies.csv
tmdb_5000_credits.csv
