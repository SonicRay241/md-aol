import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.recommender import Recommender

if (__name__ == "__main__"):
    rec_model = Recommender("./scripts/netflix_titles.csv")

    rec_model.clean_data()
    rec_model.create_features()
    rec_model.train()
    
    print(rec_model.recommend("naruto"))

    rec_model.save("./model/model_bundle.pkl")