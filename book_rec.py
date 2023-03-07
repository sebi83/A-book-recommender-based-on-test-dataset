import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# load ratings
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';',error_bad_lines=False, dtype={'ISBN': str})
ratings = ratings[ratings['Book-Rating']!=0]
print(ratings.shape)

# load books
books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';',error_bad_lines=False, low_memory=False, dtype={'ISBN': str})

# merge ratings and books data
dataset = pd.merge(ratings, books, on=['ISBN'])

# drop any rows with missing values
dataset.dropna(inplace=True)

# convert all string columns to lowercase
dataset_lowercase = dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

# define function to get similar books
def get_similar_books(book_title):
    # get readers who have read the selected book
    similar_readers = dataset_lowercase['User-ID'][(dataset_lowercase['Book-Title']==book_title)]
    similar_readers = similar_readers.tolist()
    similar_readers = np.unique(similar_readers)

    # filter dataset to books read by similar readers
    books_of_similar_readers = dataset_lowercase[(dataset_lowercase['User-ID'].isin(similar_readers))]

    # group by book title and count number of ratings
    number_of_ratings_per_book = books_of_similar_readers.groupby(['Book-Title']).agg('count').reset_index()

    # select books with more than 8 ratings
    books_to_compare = number_of_ratings_per_book['Book-Title'][number_of_ratings_per_book['User-ID'] >= 8]
    books_to_compare = books_to_compare.tolist()

    # get rating data for selected books
    ratings_data_raw = books_of_similar_readers[['User-ID', 'Book-Rating', 'Book-Title']][books_of_similar_readers['Book-Title'].isin(books_to_compare)]

    # group by user and book, and compute mean rating
    ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

    # reset index to see User-ID in every row
    ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

    # create pivot table of user ratings for each book
    dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

    # compute correlation with selected book for all other books
    correlations = dataset_for_corr.corrwith(dataset_for_corr[book_title])

    # get top 10 similar books
    similar_books = correlations.sort_values(ascending=False)[1:11]

    return similar_books.index.tolist()

def app():
    st.title('Book Recommendation System')

    # select book to get similar books
    book_options = books['Book-Title'].sample(n=100).tolist()
    selected_book = st.text_input('Enter a book title to get similar books', '')

    # check if input is valid and get similar books
    if selected_book:
        # remove any leading/trailing spaces
        selected_book = selected_book.strip()

        # check if input is in the list of available books
        if selected_book in book_options:
            # get similar books and display results
            similar_books = get_similar_books(selected_book)
            st.write('Top 10 books similar to', selected_book)
            for book in similar_books:
                st.write(book)
        else:
            st.write('Sorry, this book is not available in our database.')
    else:
        st.write('Please enter a book title')

if __name__ == '__main__':
    app()
