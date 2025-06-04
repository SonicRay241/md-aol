from fastapi import FastAPI
import joblib
import requests
import dotenv
import os

from model.recommender import Recommender

# Load Env
dotenv.load_dotenv()
model_url = os.getenv("MODEL_URL")
file_path = "./model/model_bundle.pkl"

# Download the file if not available
if not os.path.isfile(file_path):
    response = requests.get(model_url)

    with open(file_path, 'wb') as f:
        f.write(response.content)

# Unpack
model_bundle: Recommender = Recommender.load(file_path)

# Init
app = FastAPI()

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
    rec = model_bundle.recommend(title, top_n=5, genre=genre, show_type=show_type, country=country)
    
    return {
        "results": rec[['title', 'type', 'listed_in', 'country']]
    }