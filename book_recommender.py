import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import streamlit as st

# load ratings and books data
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';', error_bad_lines=False, low_memory=False)
ratings = ratings[ratings['Book-Rating']!=0] #no zero ratings

# load books
books = pd.read_csv('BX-Books.csv',  encoding='cp1251', sep=';', error_bad_lines=False, low_memory=False)

# lowercase all string columns in both datasets
ratings = ratings.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)
books = books.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

# merge ratings and books data on ISBN
dataset = pd.merge(ratings, books, on='ISBN')

# get a list of unique readers who read a particular book
def get_readers(book_title):
    return dataset['User-ID'][(dataset['Book-Title'] == book_title)].unique()

# get books read by a list of readers
def get_books_by_readers(reader_list):
    return dataset[(dataset['User-ID'].isin(reader_list))]

# get number of ratings for each book read by a list of readers
def get_rating_counts(df, threshold):
    book_counts = df.groupby(['Book-Title']).agg('count').reset_index()
    return book_counts['Book-Title'][book_counts['User-ID'] >= threshold].tolist()

# get mean rating for each reader and book pair
def get_mean_ratings(df):
    return df.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean().to_frame().reset_index()

# pivot mean ratings to make each book a column
def pivot_ratings(df):
    return df.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

# compute correlations between a book and all other books
def get_book_correlations(df, book_title):
    other_books = df.drop([book_title], axis=1)
    book_titles = []
    correlations = []
    avg_ratings = []

    for title in list(other_books.columns.values):
        book_titles.append(title)
        correlations.append(df[book_title].corr(df[title]))
        avg_ratings.append(get_mean_ratings(dataset[(dataset['Book-Title'] == title)])['Book-Rating'].min())

    return pd.DataFrame(list(zip(book_titles, correlations, avg_ratings)), columns=['book', 'corr', 'avg_rating'])

# main function to compute correlations for a list of books
def compute_book_correlations(book_list, threshold):
    results = []

    for book_title in book_list:
        readers = get_readers(book_title)
        books_by_readers = get_books_by_readers(readers)
        relevant_books = get_rating_counts(books_by_readers, threshold)
        ratings_data = get_mean_ratings(books_by_readers[books_by_readers['Book-Title'].isin(relevant_books)])
        pivoted_ratings = pivot_ratings(ratings_data)
        correlations = get_book_correlations(pivoted_ratings, book_title)
        results.append(correlations.sort_values('corr', ascending=False).head(10))

    return results

# Streamlit app
st.title('Book Recommender')

# get user input
search_term = st.text_input('Search for a book')

if search_term:
    # find books with similar titles to user input
    matches = process.extract(search_term, books['Book-Title'].unique(), scorer=fuzz.token_sort_ratio)
    book_matches = [match[0] for match in matches if match[1] >= 75]

    if len(book_matches) == 0:
        st.write(f"No books found matching '{search_term}'")
    else:
        st.write(f"Found {len(book_matches)} books matching '{search_term}'")

        # get user's selected book
        selected_book = st.selectbox("Select a book", book_matches)

        # get book correlations
        threshold = st.slider("Minimum number of ratings per book", min_value=1, max_value=100, step=1, value=20)
        st.write(f"Computing correlations for '{selected_book}' with at least {threshold} ratings")
        correlations = compute_book_correlations([selected_book], threshold)[0]

    # display results
    st.write(f"\nTop 10 book recommendations for '{selected_book}':\n")
    for _, row in correlations.iterrows():
        st.write(f"{row['book']}: correlation={row['corr']:.2f}, average rating={row['avg_rating']:.2f}")

