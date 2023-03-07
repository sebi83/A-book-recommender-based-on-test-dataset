from fuzzywuzzy import fuzz
import streamlit as st
import pandas as pd
import numpy as np

# Load data
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';', low_memory=False, on_bad_lines='skip')
books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';', on_bad_lines='skip', low_memory=False)

# lowercase book titles and authors
dataset = pd.merge(ratings, books, on=['ISBN'])
dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

# Function to generate book recommendations based on input book
def find_closest_match(book_title):
    book_titles = dataset['Book-Title'].unique()
    matches = []
    for title in book_titles:
        ratio = fuzz.ratio(book_title, title.lower())
        if ratio >= 70:
            matches.append(title)
    return matches

# Function to generate book recommendations based on input book
def generate_recommendations(book_title):
    # Check if input book exists in the dataset
    book_title = book_title.lower()
    if book_title not in dataset['Book-Title'].str.lower().unique():
        matches = find_closest_match(book_title)
        if len(matches) > 0:
            book_title = matches[0]
            st.warning(f"Book not found in the dataset. Did you mean {book_title}?")
        else:
            st.warning("Book not found in the dataset.")
            return []
    
    # Get readers who read input book
    readers = dataset.loc[dataset['Book-Title'].str.lower() == book_title, 'User-ID'].unique()

    # Check if there are any readers who read the input book
    if len(readers) == 0:
        st.warning("Book not found in the dataset.")
        return []


    # Get data for readers who read input book
    books_readers = dataset.loc[dataset['User-ID'].isin(readers)]

    # Number of ratings per book
    book_ratings = books_readers.groupby(['Book-Title']).agg('count').reset_index()

    # Select books with high enough number of ratings
    books_to_compare = book_ratings.loc[book_ratings['User-ID'] >= 8, 'Book-Title'].tolist()

    # Get rating data for selected books
    ratings_data_raw = books_readers.loc[books_readers['Book-Title'].isin(books_to_compare), ['User-ID', 'Book-Rating', 'Book-Title']]

    # Compute mean ratings for each book
    ratings_data_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()
    ratings_data_nodup = ratings_data_nodup.to_frame().reset_index()

    # Create correlation matrix for books
    book_matrix = ratings_data_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

    # Compute correlation between input book and other books
    book_corr = book_matrix.corrwith(book_matrix[book_title])

    # Combine correlation and average rating into dataframe
    corr_df = pd.DataFrame({'Correlation': book_corr, 'Avg Rating': book_matrix.mean()}, index=book_matrix.columns)

    # Filter out books with too few ratings
    corr_df = corr_df.dropna()
    corr_df = corr_df[corr_df['Avg Rating'] > 8]

    # Sort by correlation and average rating
    corr_df = corr_df.sort_values(['Correlation', 'Avg Rating'], ascending=[False, False])

    # Return top 10 recommendations
    return corr_df.sort_values(['Correlation', 'Avg Rating'], ascending=[False, False]).head(10).index.tolist()


# Streamlit app
with st.spinner("Loading data..."):
    st.title("Book Recommendation System")

book_input = st.text_input("Enter your favorite book:", "")

if st.button("Generate Recommendations"):
    # Show loading spinner
    with st.spinner("Generating recommendations..."):
        # Find closest match to input book title
        closest_matches = find_closest_match(book_input)
        if len(closest_matches) > 0:
            st.warning(f"Book not found in the dataset. Did you mean one of these? {', '.join(closest_matches)}")
        else:
            # Generate recommendations
            recommendations = generate_recommendations(book_input)

            # Display recommendations
            st.subheader("Top 10 Recommended Books:")
            if len(recommendations) == 0:
                st.warning("No recommendations found.")
            else:
                for i, book in enumerate(recommendations):
                    st.write(f"{i+1}. {book}")
