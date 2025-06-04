import joblib

from lib import Recommender

file_path = "./model/model_bundle.pkl"
model_bundle = Recommender.load(file_path)

print(model_bundle.recommend("breaking bad", top_n=2))