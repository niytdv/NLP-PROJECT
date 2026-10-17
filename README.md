# The Movie Cinema

![Python](https://img.shields.io/badge/Python-3.8+-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)
![API](https://img.shields.io/badge/API-TMDB-fcba03)
![NLP](https://img.shields.io/badge/NLP-Naive%20Bayes-orange)
![ML](https://img.shields.io/badge/ML-scikit--learn-blue)

A movie recommendation and sentiment analysis web app. Search for any movie, get its details, see who's in it, get similar recommendations — and read real user reviews with each one automatically labelled as Positive or Negative by an NLP model.

---

## How It Works

The app has three core capabilities working together: movie data retrieval, content-based recommendation, and NLP-driven sentiment analysis.

### 1. Movie Data — TMDB API

All movie metadata (title, overview, genres, release date, runtime, rating, poster, cast) is fetched from [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api) entirely client-side in JavaScript. This includes:

- Movie search and details via `/movie/{id}`
- Top cast via `/movie/{id}/credits`
- Individual cast details via `/person/{id}`
- Recommended movies via `/movie/{id}/recommendations`
- User reviews via `/movie/{id}/reviews`

### 2. Content-Based Movie Recommendation

The recommendation engine runs on the server side in Flask. It uses a pre-processed dataset (`main_data.csv`) built from multiple Kaggle sources that contains ~10,000 movies with metadata like genres, keywords, cast, director, and taglines combined into a single feature string per movie.

**How similarity is computed:**

1. Each movie's combined feature string is vectorized using `CountVectorizer` (bag-of-words).
2. Cosine similarity is computed across all movie vectors.
3. When a movie is selected, the top N most similar movies are returned as recommendations.

```
feature string = genres + keywords + tagline + cast + director
CountVectorizer → sparse matrix → cosine_similarity → top N matches
```

This is a classic content-based filtering approach — no user history or collaborative signals needed.

### 3. NLP Sentiment Analysis

This is the core NLP component of the project. Every user review fetched from TMDB is automatically classified as **Positive 😃** or **Negative 😔**.

#### Pipeline

```
Review text (raw string)
    ↓
TF-IDF Vectorizer
  - Lowercasing
  - ASCII accent stripping
  - English stop word removal
  - IDF weighting
    ↓
Multinomial Naive Bayes Classifier
    ↓
Positive / Negative label
```

#### Model Details

| Property | Value |
|---|---|
| Algorithm | Multinomial Naive Bayes |
| Vectorizer | TF-IDF (`TfidfVectorizer`) |
| Training data | ~6,900 labelled movie reviews (`datasets/reviews.txt`) |
| Test accuracy | ~97.5% (80/20 split, `random_state=42`) |
| Saved as | `nlp_model.pkl`, `tranform.pkl` |

The model is trained on a dataset of real movie reviews labelled `1` (positive) or `0` (negative). TF-IDF converts each review into a weighted term-frequency vector, downweighting common words that appear across many documents. Multinomial Naive Bayes then applies Bayes' theorem over word feature probabilities to classify the sentiment.

#### Why Naive Bayes for text?

- Works well on sparse, high-dimensional data like TF-IDF vectors
- Fast to train and predict
- Strong baseline for short text classification
- Achieves near state-of-the-art on sentiment tasks with limited data

---

## Architecture

```
Browser (JS)
    │
    ├─ TMDB API calls (search, cast, recommendations, reviews)
    │       ↓
    │   reviews collected as JSON array
    │
    └─ POST /recommend  ──────────────────────────────────────────────┐
            (movie metadata + reviews JSON)                           │
                                                                      ▼
                                                              Flask (main.py)
                                                                      │
                                                     ┌────────────────┴──────────────────┐
                                                     │                                   │
                                              Recommendation                     Sentiment Analysis
                                              Engine                             (NLP pipeline)
                                              (cosine similarity                 TF-IDF → Naive Bayes
                                               on main_data.csv)                 per review
                                                     │                                   │
                                                     └────────────────┬──────────────────┘
                                                                      │
                                                              render recommend.html
                                                              (movie info + cast +
                                                               recommendations +
                                                               reviews with labels)
```

---

## Project Structure

```
├── main.py                  # Flask app — recommendation + sentiment logic
├── retrain_model.py         # Retrain NLP model (run after upgrading scikit-learn)
├── nlp_model.pkl            # Trained Naive Bayes classifier
├── tranform.pkl             # Fitted TF-IDF vectorizer
├── main_data.csv            # Processed movie dataset for recommendations
├── datasets/
│   ├── reviews.txt          # Labelled review dataset for training (6,918 rows)
│   ├── final_data.csv       # Cleaned movie metadata
│   └── movie_metadata.csv   # Raw movie metadata
├── static/
│   ├── recommend.js         # All TMDB API calls + review fetching
│   ├── autocomplete.js      # Movie title autocomplete
│   └── style.css
└── templates/
    ├── home.html            # Search page
    └── recommend.html       # Results page (movie details + sentiment table)
```

---

## Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/niytdv/NLP-PROJECT.git
cd NLP-PROJECT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a TMDB API key

Create a free account at [themoviedb.org](https://www.themoviedb.org/), go to **Settings → API**, and request an API key. If asked for a website URL, enter `NA`.

### 4. Add your API key

Open `static/recommend.js` and replace `YOUR_API_KEY` on line 2:

```javascript
var myAPI = 'YOUR_API_KEY'
```

### 5. Retrain the NLP model (if needed)

The `.pkl` files must match your installed scikit-learn version. If you see a `NotFittedError` or `InconsistentVersionWarning` on startup, regenerate them:

```bash
python retrain_model.py
```

### 6. Run

```bash
python main.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## A Note on IMDB Scraping

The original project scraped user reviews from IMDB using `beautifulsoup4`. This no longer works for two reasons:

- **2023** — IMDB rewrote their frontend in React/Next.js. The old HTML class `text show-more__control` no longer exists in the DOM.
- **2024–2026** — IMDB deployed AWS WAF bot protection. All server-side HTTP requests receive a `202` challenge response with an empty body. There is no workaround without a paid residential proxy service.

Reviews are now fetched from **TMDB's own reviews API** (`/movie/{id}/reviews`) client-side in JavaScript — same API key, no scraping, no bot detection.

---

## Datasets

| Dataset | Source |
|---|---|
| IMDB 5000 Movie Dataset | [Kaggle](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset) |
| The Movies Dataset | [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset) |
| Movies of 2018 | [Wikipedia](https://en.wikipedia.org/wiki/List_of_American_films_of_2018) |
| Movies of 2019 | [Wikipedia](https://en.wikipedia.org/wiki/List_of_American_films_of_2019) |
| Movies of 2020 | [Wikipedia](https://en.wikipedia.org/wiki/List_of_American_films_of_2020) |

---

Please do ⭐ the repository if it helped you in any way.
