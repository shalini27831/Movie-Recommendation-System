import streamlit as st
import pandas as pd
import ast
import os
import base64
from html import escape

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM CSS  (theme: night at the cinema)
# Palette: ink #14101c · velvet #1f1a2b · marquee gold #f2b33d
#          curtain rose #e5486a · paper #f3eee6
# -------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --ink: #14101c;
    --velvet: #1f1a2b;
    --velvet-2: #2a2338;
    --line: #3a3150;
    --gold: #f2b33d;
    --rose: #e5486a;
    --paper: #f3eee6;
    --muted: #a79fb8;
}

/* ---------- base ---------- */

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: var(--paper);
}

.stApp {
    background:
        radial-gradient(1100px 500px at 50% -120px, rgba(229,72,106,0.22), transparent 70%),
        radial-gradient(800px 400px at 100% 0%, rgba(242,179,61,0.10), transparent 70%),
        var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1200px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    font-family: 'Fraunces', serif;
    color: var(--paper);
    letter-spacing: -0.01em;
}

/* ---------- film strip + hero ---------- */

.filmstrip {
    height: 18px;
    border-radius: 4px;
    margin-bottom: 28px;
    background:
        repeating-linear-gradient(
            90deg,
            var(--ink) 0 14px,
            transparent 14px 28px
        ) center / 100% 8px no-repeat,
        #000;
    border: 1px solid var(--line);
}

.title {
    text-align: center;
    font-family: 'Fraunces', serif;
    font-size: 54px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--paper);
    margin-bottom: 10px;
    text-shadow: 0 0 28px rgba(242,179,61,0.35);
}

.subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 18px;
    max-width: 640px;
    margin: 0 auto 40px auto;
}

/* ---------- selector panel ---------- */

.stSelectbox label p {
    color: var(--muted);
    font-size: 16px;
    font-weight: 500;
}

div[data-baseweb="select"] > div {
    font-size: 17px;
    background-color: var(--velvet);
    border: 1px solid var(--line);
    border-radius: 12px;
    min-height: 52px;
    color: var(--paper);
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--gold);
}

ul[role="listbox"] {
    background-color: var(--velvet);
}

/* ---------- button ---------- */

div.stButton > button {
    background: linear-gradient(90deg, var(--rose), #f06a4d);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 24px rgba(229,72,106,0.30);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

div.stButton > button p {
    font-size: 18px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #f0587a, #f77b5c);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(229,72,106,0.42);
}

div.stButton > button:focus-visible {
    outline: 3px solid var(--gold);
    outline-offset: 2px;
}

/* ---------- section headings ---------- */

.section-title {
    font-family: 'Fraunces', serif;
    font-size: 26px;
    font-weight: 600;
    margin: 34px 0 6px 0;
    color: var(--paper);
}

.section-note {
    color: var(--muted);
    margin-bottom: 18px;
}

.picked {
    color: var(--gold);
}

/* ---------- fixed poster ---------- */

.fixed-poster {
    text-align: center;
    margin: 0 0 22px 0;
}

.fixed-poster img {
    width: 220px;
    max-width: 80%;
    border-radius: 14px;
    border: 1px solid var(--line);
    box-shadow: 0 14px 40px rgba(0,0,0,0.55), 0 0 40px rgba(242,179,61,0.12);
}

/* ---------- movie cards ---------- */

.movie-card {
    background: linear-gradient(160deg, var(--velvet-2), var(--velvet));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px 18px 18px 18px;
    margin-top: 10px;
    margin-bottom: 20px;
    height: 190px;
    box-sizing: border-box;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, border-color 0.25s ease;
}

.movie-card:hover {
    transform: translateY(-6px);
    border-color: var(--gold);
}

.movie-rank {
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 800;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 10px;
}

.movie-title {
    font-family: 'Fraunces', serif;
    font-size: 19px;
    font-weight: 600;
    line-height: 1.25;
    color: var(--paper);
}

/* ---------- how it works ---------- */

.step-card {
    background: var(--velvet);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 22px 20px;
    height: 200px;
}

.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--gold);
    color: var(--ink);
    font-weight: 700;
    margin-bottom: 12px;
}

.step-title {
    font-family: 'Fraunces', serif;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--paper);
}

.step-text {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.55;
}

hr {
    border-color: var(--line);
    margin-top: 2.5rem;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 640px) {
    .title { font-size: 36px; }
    .step-card { height: auto; }
    .movie-card { height: auto; min-height: 120px; }
}

@media (prefers-reduced-motion: reduce) {
    .movie-card, div.stButton > button { transition: none; }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge datasets
    movies = movies.merge(credits, on="title")

    # Select required columns
    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ]

    # Data cleaning
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)

    return movies


# -------------------------------------------------
# CONVERT JSON-LIKE COLUMNS
# -------------------------------------------------

def convert_text(text):

    try:
        data = ast.literal_eval(text)

        if isinstance(data, list):

            names = []

            for item in data:

                if isinstance(item, dict):

                    if "name" in item:
                        names.append(item["name"])

            return " ".join(names)

        return str(text)

    except:
        return str(text)


# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------

@st.cache_data
def prepare_data(movies):

    movies = movies.copy()

    movies["genres"] = movies["genres"].apply(convert_text)
    movies["keywords"] = movies["keywords"].apply(convert_text)

    # Extract first few cast members
    def get_cast(text):

        try:
            data = ast.literal_eval(text)

            return " ".join(
                [item["name"] for item in data[:5]
                 if isinstance(item, dict) and "name" in item]
            )

        except:
            return ""

    movies["cast"] = movies["cast"].apply(get_cast)

    # Extract director from crew
    def get_director(text):

        try:
            data = ast.literal_eval(text)

            for item in data:

                if (
                    isinstance(item, dict)
                    and item.get("job") == "Director"
                ):
                    return item.get("name", "")

            return ""

        except:
            return ""

    movies["crew"] = movies["crew"].apply(get_director)

    # Same concept as notebook:
    # overview + genres + keywords + cast + crew

    movies["tags"] = (
        movies["overview"] + " "
        + movies["genres"] + " "
        + movies["keywords"] + " "
        + movies["cast"] + " "
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].str.lower()

    return movies


# -------------------------------------------------
# CREATE VECTORS
# -------------------------------------------------

@st.cache_resource
def create_vectors(tags):

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(tags)

    return cv, vectors


# -------------------------------------------------
# FIXED POSTER (display only)
# A built-in poster is shown by default. To use your own, put an
# image named poster.jpg (or .jpeg / .png / .webp / .svg) in the
# same folder as this file.
# POSTER_POSITION: "above" (under the header) or "below" (page bottom)
# -------------------------------------------------

POSTER_POSITION = "above"

DEFAULT_POSTER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 600" width="400" height="600">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2d1735"/>
      <stop offset="1" stop-color="#14101c"/>
    </linearGradient>
    <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f2b33d" stop-opacity="0.42"/>
      <stop offset="1" stop-color="#f2b33d" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="curtain" x1="0" y1="0" x2="0.34" y2="0" spreadMethod="repeat">
      <stop offset="0" stop-color="#5c1530"/>
      <stop offset="0.5" stop-color="#a52848"/>
      <stop offset="1" stop-color="#5c1530"/>
    </linearGradient>
    <clipPath id="barClip"><rect x="-110" y="0" width="220" height="34" rx="6"/></clipPath>
  </defs>

  <rect width="400" height="600" fill="url(#bg)"/>

  <polygon points="200,30 105,486 295,486" fill="url(#beam)"/>
  <ellipse cx="200" cy="478" rx="105" ry="14" fill="#f2b33d" opacity="0.16"/>

  <!-- film strip -->
  <rect x="0" y="480" width="400" height="66" fill="#0e0a14" stroke="#3a3150"/>
  <rect x="14" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="14" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="38" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="38" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="62" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="62" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="86" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="86" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="110" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="110" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="134" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="134" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="158" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="158" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="182" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="182" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="206" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="206" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="230" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="230" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="254" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="254" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="278" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="278" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="302" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="302" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="326" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="326" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="350" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="350" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="374" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="374" y="534" width="10" height="8" rx="2" fill="#14101c"/><rect x="398" y="486" width="10" height="8" rx="2" fill="#14101c"/><rect x="398" y="534" width="10" height="8" rx="2" fill="#14101c"/>
  <rect x="112" y="494" width="84" height="38" rx="3" fill="#2a2338" stroke="#3a3150"/>
  <rect x="204" y="494" width="84" height="38" rx="3" fill="#2a2338" stroke="#3a3150"/>
  <rect x="296" y="494" width="84" height="38" rx="3" fill="#2a2338" stroke="#3a3150"/>
  <rect x="20" y="494" width="84" height="38" rx="3" fill="#2a2338" stroke="#3a3150"/>

  <!-- curtains -->
  <path d="M0 0 H100 C72 150 112 300 62 600 H0 Z" fill="url(#curtain)"/>
  <path d="M400 0 H300 C328 150 288 300 338 600 H400 Z" fill="url(#curtain)"/>
  <rect x="0" y="0" width="400" height="26" fill="#5c1530"/>
  <rect x="0" y="26" width="400" height="4" fill="#f2b33d"/>

  <!-- title -->
  <text x="200" y="104" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif"
        font-size="46" font-weight="700" fill="#f3eee6">Movie Night</text>
  <text x="200" y="134" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif"
        font-size="16" font-style="italic" fill="#f2b33d">Find your next favorite</text>

  <!-- clapperboard -->
  <g transform="translate(200 332)">
    <rect x="-110" y="-40" width="220" height="150" rx="10" fill="#1f1a2b" stroke="#f2b33d" stroke-width="3"/>
    <rect x="-90" y="-8" width="180" height="2" fill="#3a3150"/>
    <rect x="-90" y="34" width="180" height="2" fill="#3a3150"/>
    <rect x="-90" y="76" width="180" height="2" fill="#3a3150"/>
    <text x="-90" y="-16" font-family="Georgia, serif" font-size="15" fill="#a79fb8">Scene 1</text>
    <text x="-90" y="26" font-family="Georgia, serif" font-size="15" fill="#a79fb8">Take 1</text>
    <text x="-90" y="68" font-family="Georgia, serif" font-size="15" fill="#a79fb8">Action!</text>

    <g transform="translate(0 -74)">
      <rect x="-110" y="0" width="220" height="34" rx="6" fill="#14101c" stroke="#f2b33d" stroke-width="3"/>
      <g clip-path="url(#barClip)">
      <polygon points="-92,0 -70,0 -84,34 -106,34" fill="#f2b33d"/>
      <polygon points="-48,0 -26,0 -40,34 -62,34" fill="#f2b33d"/>
      <polygon points="-4,0 18,0 4,34 -18,34" fill="#f2b33d"/>
      <polygon points="40,0 62,0 48,34 26,34" fill="#f2b33d"/>
      <polygon points="84,0 106,0 92,34 70,34" fill="#f2b33d"/>
      <polygon points="128,0 150,0 136,34 114,34" fill="#f2b33d"/>
      </g>
    </g>
    <g transform="rotate(-16 -110 -40) translate(0 -112)">
      <rect x="-110" y="0" width="220" height="34" rx="6" fill="#14101c" stroke="#f2b33d" stroke-width="3"/>
      <g clip-path="url(#barClip)">
      <polygon points="-92,0 -70,0 -84,34 -106,34" fill="#f2b33d"/>
      <polygon points="-48,0 -26,0 -40,34 -62,34" fill="#f2b33d"/>
      <polygon points="-4,0 18,0 4,34 -18,34" fill="#f2b33d"/>
      <polygon points="40,0 62,0 48,34 26,34" fill="#f2b33d"/>
      <polygon points="84,0 106,0 92,34 70,34" fill="#f2b33d"/>
      <polygon points="128,0 150,0 136,34 114,34" fill="#f2b33d"/>
      </g>
    </g>
    <circle cx="-110" cy="-40" r="6" fill="#f2b33d"/>
  </g>

  <text x="200" y="578" text-anchor="middle" font-family="Georgia, serif" font-size="14" fill="#a79fb8">Pick a film. Get five more.</text>
</svg>
"""


@st.cache_data(show_spinner=False)
def load_fixed_poster():

    base = os.path.dirname(os.path.abspath(__file__))

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }

    for ext, mime in mime_types.items():

        path = os.path.join(base, f"poster{ext}")

        if os.path.exists(path):

            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            return f"data:{mime};base64,{encoded}"

    # No poster file found: use the built-in poster
    encoded = base64.b64encode(DEFAULT_POSTER_SVG.encode()).decode()

    return f"data:image/svg+xml;base64,{encoded}"


def show_fixed_poster():

    poster = load_fixed_poster()

    if poster:
        st.markdown(
            f'<div class="fixed-poster">'
            f'<img src="{poster}" alt="Featured movie poster">'
            f'</div>',
            unsafe_allow_html=True
        )


# -------------------------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------------------------

def recommend(movie_title, movies, vectors):

    matching_movies = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if matching_movies.empty:
        return []

    movie_index = matching_movies.index[0]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    # Sort similarity scores
    movie_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in movie_indices:

        # Skip selected movie
        if index == movie_index:
            continue

        recommendations.append(
            (
                movies.iloc[index]["title"],
                similarity_scores[index]
            )
        )

        if len(recommendations) == 5:
            break

    return recommendations


# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------

movies = load_data()
movies = prepare_data(movies)

cv, vectors = create_vectors(movies["tags"])


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies similar to your favorite movie using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# FIXED POSTER (above)
# -------------------------------------------------

if POSTER_POSITION == "above":
    show_fixed_poster()


# -------------------------------------------------
# MOVIE SELECTION
# -------------------------------------------------

st.markdown(
    '<div class="section-title">🔎 Select a Movie</div>',
    unsafe_allow_html=True
)

movie_list = sorted(movies["title"].unique())

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)


# -------------------------------------------------
# RECOMMEND BUTTON
# -------------------------------------------------

if st.button("🍿 Recommend Movies", use_container_width=True):

    recommendations = recommend(
        selected_movie,
        movies,
        vectors
    )

    st.markdown(
        f'<div class="section-title">Recommended Movies for '
        f'<span class="picked">{escape(selected_movie)}</span></div>',
        unsafe_allow_html=True
    )

    if recommendations:

        cols = st.columns(5)

        for rank, (col, (title, score)) in enumerate(
            zip(cols, recommendations), start=1
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-rank">{rank}</div>
                        <div class="movie-title">{escape(title)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.warning("No recommendations found.")


# -------------------------------------------------
# PROJECT INFORMATION
# -------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 How It Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Feature Engineering</div>
            <div class="step-text">
                Movie overview, genres, keywords, cast and director
                are combined into a single text feature.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Text Vectorization</div>
            <div class="step-text">
                CountVectorizer converts the movie tags into numerical
                feature vectors.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Similarity</div>
            <div class="step-text">
                Cosine similarity compares movies and identifies
                the five most similar movies.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# FIXED POSTER (below)
# -------------------------------------------------

if POSTER_POSITION == "below":
    show_fixed_poster()