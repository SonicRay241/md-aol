from fastapi import FastAPI
from contextlib import asynccontextmanager
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

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    file_path = "./model/model_bundle.pkl"

    # Download the file if not available
    if not os.path.isfile(file_path):
        file_path = download_model(model_url)

    # Unpack
    models["recommender"] = Recommender.load(file_path)

    yield
    # Clean up the ML models and release the resources
    models.clear()

# Init
app = FastAPI(lifespan=lifespan)

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
    rec = models["recommender"].recommend(title, top_n=5, genre=genre, show_type=show_type, country=country)

    return {
        "results": rec[['title', 'type', 'listed_in', 'country']]
    }