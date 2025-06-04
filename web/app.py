import streamlit as st
import requests
import os
import dotenv
import pycountry
import pandas as pd

countries = map(lambda country: country.name, pycountry.countries)
dotenv.load_dotenv()
api_url = os.getenv("API_URL")

st.title("Netflix Recommendation Model")

search = st.text_input("Search Titles", placeholder="Enter Title Here")

st.subheader("Filters")

filter1, filter2, filter3 = st.columns(3)

search_btn = st.button("Search")
show_type = filter1.selectbox(
    "Show Type",
    ("TV Show", "Movie"),
    index=None,
    placeholder="Select Show Type"
)

genre = filter2.text_input("Genre", placeholder="Type genre...")

country = filter3.selectbox(
    "Country",
    countries,
    index=None,
    placeholder="Select Country"
)

params = {
    "title": search
}

if search_btn:
    with st.spinner("Searching...", show_time = True):
        if show_type: params["show_type"] = show_type
        if genre: params["genre"] = genre
        if country: params["country"] = country

        response = requests.get(api_url + "/search", params=params)

        res = response.json()

        if res["results"]:
            recs = pd.DataFrame(data=res["results"])
            recs = recs.replace("", "-", regex=False)
            st.subheader("Results")
            st.dataframe(
                recs,
                hide_index=True,
                column_config={
                    "title": "Title",
                    "type": "Show Type",
                    "listed_in": "Genre",
                    "country": "Country"
                }
            )

        else:
            st.error("No results", icon="😔")
