# The-Movie-Cinema

![Python](https://img.shields.io/badge/Python-3.8+-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)
![API](https://img.shields.io/badge/API-TMDB-fcba03)

This application provides all the details of the requested movie such as overview, genre, release date, rating, runtime, top cast, reviews, recommended movies, etc.

The details of the movies (title, genre, runtime, rating, poster, etc.) are fetched using the [TMDB API](https://www.themoviedb.org/documentation/api). User reviews are also fetched from TMDB's reviews API and passed through a Naive Bayes sentiment classifier to label them as Positive or Negative.

## Link to the application

Check out the live demo: https://tmc.kishanlal.dev/

If you can't find the movie you're searching for through auto-suggestions while typing, there's no need to worry. Simply type the name of the movie and press "enter". Even if you make some typos, it should still work fine.

## 'Invalid Request' Error

If you're getting an invalid request error, kindly go through this issue — https://github.com/kishan0725/The-Movie-Cinema/issues/2

## How to get the API key?

Create an account at https://www.themoviedb.org/, click on the `API` link from the left hand sidebar in your account settings and fill in the details to apply for an API key. If you are asked for a website URL, just enter "NA" if you don't have one. The API key will appear in your `API` sidebar once your request is approved.

## How to run the project?

1. Clone this repository to your local system.
2. Install all the libraries listed in [requirements.txt](https://github.com/kishan0725/The-Movie-Cinema/blob/master/requirements.txt) with:
   ```
   pip install -r requirements.txt
   ```
3. Replace `YOUR_API_KEY` at line 2 of `static/recommend.js` with your TMDB API key.
4. Retrain the sentiment model (required if your scikit-learn version differs from 0.23.1):
   ```
   python retrain_model.py
   ```
   This regenerates `nlp_model.pkl` and `tranform.pkl` compatible with your installed scikit-learn version.
5. Run the app from your project directory:
   ```
   python main.py
   ```
6. Open your browser and go to `http://127.0.0.1:5000/`.

## Notes on Reviews & Sentiment Analysis

### Why IMDB scraping was dropped
The original project scraped user reviews from IMDB using `beautifulsoup4`. This broke in two stages:

1. **2023 redesign** — IMDB rewrote their frontend in React/Next.js. The old HTML class `text show-more__control` no longer exists.
2. **AWS WAF (2024–2026)** — IMDB added AWS WAF bot protection. All server-side HTTP requests now receive a `202` challenge response with an empty body, making scraping completely impossible without a paid proxy/unblocker service.

### Current approach — TMDB Reviews API
Reviews are now fetched **client-side** from TMDB's own reviews endpoint:
```
GET https://api.themoviedb.org/3/movie/{movie_id}/reviews
```
This uses the same API key already configured in `recommend.js`. Up to 40 reviews (2 pages) are fetched in the browser and sent to Flask for sentiment analysis — no scraping, no bot challenges, no extra dependencies.

### Sentiment Analysis
Reviews are classified as **Positive** or **Negative** using a Multinomial Naive Bayes model trained on ~6,900 labelled movie reviews. The model achieves ~97% accuracy on a held-out test set.

The model is saved as:
- `nlp_model.pkl` — the trained classifier
- `tranform.pkl` — the fitted TF-IDF vectorizer

If you see a `NotFittedError` or `InconsistentVersionWarning` on startup, your scikit-learn version differs from the one used to build the `.pkl` files. Regenerate them by running:
```
python retrain_model.py
```

## Sources of the datasets

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
3. [List of movies in 2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018)
4. [List of movies in 2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)
5. [List of movies in 2020](https://en.wikipedia.org/wiki/List_of_American_films_of_2020)

Please do ⭐ the repository if it helped you in any way.
