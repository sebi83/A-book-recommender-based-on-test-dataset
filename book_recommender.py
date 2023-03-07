import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz,process

# Load dataset
@st.cache_data
def load_data():
    ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';', low_memory=False, on_bad_lines='skip')
    ratings = ratings[ratings['Book-Rating'] != 0]
    books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';', low_memory=False, on_bad_lines='skip')
    dataset = pd.merge(ratings, books, on=['ISBN'])
    dataset.dropna(inplace=True)
    return dataset

dataset = load_data()

# Get user input
st.title('Book Recommendation Engine')
book_input = st.text_input('Enter the name of a book:', 'The Fellowship of the Ring')

# Prepare dataset for recommendation
dataset_lowercase = dataset.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
readers = dataset_lowercase['User-ID'][dataset_lowercase['Book-Title']==book_input.lower()]
readers = readers.tolist()
readers = np.unique(readers)

books_of_readers = dataset_lowercase[(dataset_lowercase['User-ID'].isin(readers))]
number_of_rating_per_book = books_of_readers.groupby(['Book-Title']).agg('count').reset_index()
books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= 8]
books_to_compare = books_to_compare.tolist()

ratings_data_raw = books_of_readers[['User-ID', 'Book-Rating', 'Book-Title']][books_of_readers['Book-Title'].isin(books_to_compare)]
ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()
ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()
dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

# Compute recommendations
def get_recommendations(dataset, book_title):
    book_title = book_title.lower()
    if book_title not in dataset['Book-Title'].str.lower().unique():
        close_matches = []
        for title in dataset['Book-Title'].str.lower().unique():
            ratio = fuzz.ratio(book_title, title)
            if ratio >= 70:
                close_matches.append((title, ratio))
        if len(close_matches) > 0:
            close_matches = sorted(close_matches, key=lambda x: x[1], reverse=True)
            closest_match = close_matches[0][0]
            st.write(f"Book title not found in dataset. Did you mean '{closest_match}'?")
            book_ratings = dataset[closest_match]
            similar_books = dataset.corrwith(book_ratings)
            similar_books = similar_books.dropna().sort_values(ascending=False)
            return similar_books.head(10)
        else:
            st.write("Book title not found in dataset.")
            return None
    else:
        book_ratings = dataset[book_title]
        similar_books = dataset.corrwith(book_ratings)
        similar_books = similar_books.dropna().sort_values(ascending=False)
        return similar_books.head(10)

# Compute recommendations for user input
similar_books = get_recommendations(dataset_for_corr, book_input)

if similar_books is not None:
    st.write("Books similar to " + book_input + ":")
    for title, score in similar_books.items():
        st.write("- " + title.capitalize()) 
        
else:
    st.write('Try another book title.')