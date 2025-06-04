from fastapi import FastAPI
import joblib
import requests
import dotenv
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Env
dotenv.load_dotenv()
model_url = os.getenv("MODEL_URL")

# Download the file
response = requests.get(model_url)
with open('/model/model_bundle.pkl', 'wb') as f:
    f.write(response.content)

# Unpack
model_bundle = joblib.load("./model/model_bundle.pkl")

# Init
app = FastAPI()

df = model_bundle["dataframe"]
indices = pd.Series(df.index, index=df['title'].str.lower())
cosine_sim = model_bundle["cosine_sim"]

def get_recommendations(
        title: str,
        cosine_sim = cosine_sim,
        top_n = 10,
        genre: str | None = None,
        show_type: str | None = None,
        country: str | None = None):
    title = title.lower()
    idx = indices.get(title)

    if idx is None:
        return None

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices
    movie_indices = [i[0] for i in sim_scores]
    recs = df.iloc[movie_indices]

    if genre:
        recs = recs[recs['listed_in'].str.contains(genre, case=False, na=False)]
    if show_type:
        recs = recs[recs['type'].str.lower() == show_type.lower()]
    if country:
        recs = recs[recs['country'].str.contains(country, case=False, na=False)]

    return recs.reset_index()[['title', 'type', 'listed_in', 'country']].head(top_n)

@app.get("/")
def read_root():
    return "Use /search for querying"

@app.get("/search")
def search(
    title: str,
    genre: str | None = None,
    show_type: str | None = None,
    country: str | None = None
):
    return {
        "results": get_recommendations(title, top_n=5, genre=genre, show_type=show_type, country=country)
    }