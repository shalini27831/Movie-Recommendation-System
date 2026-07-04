# 🎬 Movie Recommendation System using Machine Learning

## 📌 Project Overview
This project builds a content-based movie recommendation system using Python and Scikit-learn. It recommends similar movies based on genres, keywords, cast, crew, and movie overviews.

## 🎯 Objectives
- Load and merge movie datasets
- Clean and preprocess data
- Perform feature engineering
- Convert text into numerical vectors using CountVectorizer
- Calculate movie similarity using Cosine Similarity
- Recommend similar movies

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab

 ⚙️ Machine Learning Techniques
- Feature Engineering
- CountVectorizer
- Cosine Similarity

 ▶️ How to Run
1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install the required libraries.
3. Upload `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
4. Run all cells.
5. Call the `recommend("Movie Name")` function.

 📚 Example

python
recommend("Avatar")


Example output:

text
Recommended Movies:
Titan A.E.
John Carter
Guardians of the Galaxy
Star Trek
Avatar: The Way of Water


🚀 Future Improvements
- Build a Streamlit web app
- Add movie posters using the TMDB API
- Deploy the project online

 👩‍💻 Author
Shalini G