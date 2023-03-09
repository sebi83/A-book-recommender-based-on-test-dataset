import streamlit as st
import pandas as pd
import numpy as np

# load data

def load_data():
    ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';', on_bad_lines='skip', low_memory=False)
    ratings = ratings[ratings['Book-Rating']!=0]
    books = pd.read_csv('BX-Books.csv',  encoding='cp1251', sep=';', on_bad_lines='skip', low_memory=False)
    dataset = pd.merge(ratings, books, on=['ISBN'])
    dataset_lowercase = dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    return dataset_lowercase

# function to get top 10 recommendations
def get_recommendations(dataset, book_title):
    # Get all readers who read a book with a similar title and author
    book_readers_all = dataset['User-ID'][(dataset['Book-Title']==book_title) & (dataset['Book-Author']=='unknown')]
    book_readers_all = book_readers_all.tolist()
    book_readers_all = np.unique(book_readers_all)
    
    # Get all books read by these readers
    books_of_book_readers_all = dataset[(dataset['User-ID'].isin(book_readers_all))]
    
    # Filter books with enough ratings
    number_of_rating_per_book = books_of_book_readers_all.groupby(['Book-Title']).agg('count').reset_index()
    books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= 8]
    books_to_compare = books_to_compare.tolist()
    
    # Get ratings of books to compare
    ratings_data_raw = books_of_book_readers_all[['User-ID', 'Book-Rating', 'Book-Title']][books_of_book_readers_all['Book-Title'].isin(books_to_compare)]
    ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()
    ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()
    
    # Convert ratings data to a pivot table for computing correlations
    dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
    
    # Compute correlations between books
    if book_title in dataset_for_corr.columns:  # check if book_title is in dataset_for_corr
        correlations = dataset_for_corr.corrwith(dataset_for_corr[book_title])
        corr_df = pd.DataFrame(correlations, columns=['Correlation'])
        corr_df.dropna(inplace=True)
        corr_df = corr_df.join(ratings_data_raw_nodup.set_index('Book-Title'), on='Book-Title')
        corr_df.drop_duplicates(subset=['User-ID'], inplace=True)
        recommendations = corr_df.groupby('Book-Title').apply(lambda x: x.sort_values(by='Correlation', ascending=False).head(10))
        recommendations.reset_index(inplace=True, drop=True)
        return recommendations['Book-Title'].tolist()
    else:
        return None  # return None if book_title is not in dataset_for_corr

# create Streamlit app
write a stramlit app that show top 10 recommendations for a book title entered by the user
def main():
    st.title('Book Recommender')
    dataset = load_data()
    book_title = st.text_input('Enter a book title')
    if book_title:
        recommendations = get_recommendations(dataset, book_title)
        if recommendations:
            st.write('Top 10 recommendations for', book_title, ':')
            for i, book in enumerate(recommendations):
                st.write(i+1, book)