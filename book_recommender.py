import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import streamlit as st



# load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';', on_bad_lines='skip', low_memory=False)
    ratings = ratings[ratings['Book-Rating'] != 0]  # no zero ratings

    # load books
    books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';', on_bad_lines='skip', low_memory=False)

    # lowercase all
    ratings = ratings.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)
    books = books.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    # merge ratings and books data on ISBN
    dataset = pd.merge(ratings, books, on='ISBN')

    return dataset, books

dataset, books = load_data()

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
    if book_title not in df.columns:
        return pd.DataFrame(columns=['book', 'author', 'corr', 'avg_rating'])

    other_books = df.drop([book_title], axis=1)
    book_titles = []
    authors = []
    correlations = []
    avg_ratings = []

    for title in list(other_books.columns.values):
        if title in df.columns:
            book_titles.append(title)
            authors.append(books.loc[books['Book-Title'] == title, 'Book-Author'].iloc[0])

            # Calculate correlation only when there are common non-null elements
            common_elements = ~np.isnan(df[book_title]) & ~np.isnan(df[title])
            if np.sum(common_elements) > 0:
                correlations.append(df[book_title][common_elements].corr(df[title][common_elements]))
            else:
                correlations.append(np.nan)

            avg_ratings.append(get_mean_ratings(dataset[(dataset['Book-Title'] == title)])['Book-Rating'].min())

    return pd.DataFrame(list(zip(book_titles, authors, correlations, avg_ratings)), columns=['book', 'author', 'corr', 'avg_rating'])




def compute_book_correlations(book_list, threshold):
    results = []

    for book_title in book_list:
        readers = get_readers(book_title)
        books_by_readers = get_books_by_readers(readers)
        relevant_books = get_rating_counts(books_by_readers, threshold)
        ratings_data = get_mean_ratings(books_by_readers[books_by_readers['Book-Title'].isin(relevant_books)])
        pivoted_ratings = pivot_ratings(ratings_data)
        correlations = get_book_correlations(pivoted_ratings, book_title)
        correlations = correlations.dropna(subset=['corr'])  # Remove NaN values from the correlations DataFrame
        results.append(correlations.sort_values('corr', ascending=False).head(10))

    return results


# Streamlit
st.title('Book Recommender')

# User input
search_term = st.text_input('Search for a book')

if search_term:
    with st.spinner("Searching for matching books..."):
        unique_book_titles = set(books['Book-Title'].unique())
        matches = process.extract(search_term, unique_book_titles, scorer=fuzz.token_set_ratio, limit=None)
        book_matches = [match[0] for match in matches if match[1] >= 80]

    if len(book_matches) == 0:
        st.write(f"No books found matching '{search_term}'")
    else:
        st.write(f"Found {len(book_matches)} books matching '{search_term}'")
        selected_book = st.selectbox("Select a book", book_matches)

        if selected_book not in unique_book_titles:
            st.write(f"Error: Book '{selected_book}' not found in the database.")
        else:
            threshold = 1
            with st.spinner(f"Computing correlations for '{selected_book}' with at least {threshold} ratings..."):
                correlations = compute_book_correlations([selected_book], threshold)[0]

            if correlations.shape[0] < 10:
                st.write(f"Could not find 10 correlated books for '{selected_book}'")
            else:
                st.write(f"\nTop 10 book recommendations for '{selected_book}':\n")
                for _, row in correlations.iloc[:10].iterrows():
                    st.write(f"{row['book']} by {row['author']}: correlation={row['corr']:.2f}, average rating={row['avg_rating']:.2f}")
