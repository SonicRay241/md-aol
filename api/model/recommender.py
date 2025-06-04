import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, data_path: str):
        self.df: pd.DataFrame = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.indices: pd.Series | None = None
        self.cosine_sim = None

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.fillna('', inplace=True)

        invalid_ratings = ['74 min', '84 min', '66 min']
        self.df = self.df.drop(self.df[self.df['rating'].isin(invalid_ratings)].index)

        # Change date_added to datetime format
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], format='%B %d, %Y', errors='coerce')

        # Reset index in case it isn't sequential
        self.df.reset_index(inplace=True)
    
    def create_features(self):
        self.df['combined_features'] = (
            (self.df['title'] + ' ') * 3 +
            (self.df['director'] + ' ') +
            (self.df['cast'] + ' ') +
            (self.df['listed_in'] + ' ') * 2 +
            (self.df['description'] + ' ') +
            (self.df['country'] + ' ') +
            (self.df['rating'] + ' ')
        )

        self.df['combined_features'] = self.df['combined_features'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    def train(self):
        if 'combined_features' not in self.df.columns:
            raise KeyError('You have not initialized the features column. Please run create_features() before training.')
        
        # Apply to your combined features
        tfidf_matrix =self.vectorizer.fit_transform(self.df['combined_features'])

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create mapping from title to index
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower())

    def recommend(
        self,
        title: str,
        top_n = 10,
        genre: str | None = None,
        show_type: str | None = None,
        country: str | None = None
    ):
        title = title.lower()
        idx = self.indices.get(title)

        if idx is None:
            return None

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get indices
        movie_indices = [i[0] for i in sim_scores]
        recs = self.df.iloc[movie_indices]

        if genre:
            recs = recs[recs['listed_in'].str.contains(genre, case=False, na=False)]
        if show_type:
            recs = recs[recs['type'].str.lower() == show_type.lower()]
        if country:
            recs = recs[recs['country'].str.contains(country, case=False, na=False)]

        return recs.reset_index().head(top_n)

    @classmethod
    def load(cls, path: str):
        instance = joblib.load(path)

        if not isinstance(instance, cls):
            raise TypeError(
                f"The object loaded must be an instance of {cls.__name__}, but got an instance of {type(instance).__name__}"
            )
        
        return instance
    
    def save(self, path: str):
        joblib.dump(self, path)