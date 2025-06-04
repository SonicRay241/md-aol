from fastapi import FastAPI
import joblib
import requests
import dotenv
import os

from model.recommender import Recommender

def download_model(url: str):
    response = requests.get(model_url)

    file_data = BytesIO()
    for chunk in response.iter_content(stream=True):
        i += 1
        if chunk:  # filter out keep-alive new chunks
            file_data.write(chunk)
    
    file_data.seek(0)
    return file_data

# Load Env
dotenv.load_dotenv()
model_url = os.getenv("MODEL_URL")
file_path = "./model/model_bundle.pkl"

# Download the file if not available
if not os.path.isfile(file_path):
    file_path = download_model(model_url)

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