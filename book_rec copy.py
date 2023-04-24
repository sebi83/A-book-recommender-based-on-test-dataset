import pandas as pd
import numpy as np

def load_datasets():
    # Load datasets
    ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';')
    ratings = ratings[ratings['Book-Rating']!=0]
    books = pd.read_csv('BX-Books.csv',  encoding='cp1251', sep=';', error_bad_lines=False)
    users = pd.read_csv('BX-Users.csv', encoding='cp1251', sep=';')

    # Merge datasets
    user_ratings = pd.merge(ratings, users, on=['User-ID'])
    dataset = pd.merge(user_ratings, books, on=['ISBN'])
    return dataset

def filter_and_normalize_data(dataset):
    """Apply lowercase normalization to string columns in a pandas DataFrame.

    Args:
        dataset (pandas.DataFrame): A table of data with mixed data types.

    Returns:
        pandas.DataFrame: A copy of the input table with lowercase string columns
        and the rest of the columns unchanged.
    """
    dataset_lowercase = dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    return dataset_lowercase

def find_tolkien_readers(dataset_lowercase):
    """Return a list of unique User-IDs who have read 'The Fellowship of the Ring' by J.R.R. Tolkien.

    :param dataset_lowercase: A pandas DataFrame containing book data in lowercase format.
    :return: A list of unique User-IDs who have read 'The Fellowship of the Ring' by J.R.R. Tolkien.
    """
    tolkien_readers = dataset_lowercase['User-ID'][(dataset_lowercase['Book-Title']=='the fellowship of the ring (the lord of the rings, part 1)') & (dataset_lowercase['Book-Author'].str.contains("tolkien"))]
    tolkien_readers = tolkien_readers.tolist()
    tolkien_readers = np.unique(tolkien_readers)
    return tolkien_readers


def create_books_of_tolkien_readers(dataset_lowercase, tolkien_readers):
    """
    Extracts and returns a pandas DataFrame containing books read by readers who have read Tolkien books.

    Args:
        dataset_lowercase (pandas.DataFrame): A DataFrame containing book ratings, with columns 'User-ID',
            'ISBN', and 'Book-Rating'.
        tolkien_readers (list): A list of User-IDs who have read Tolkien books.

    Returns:
        pandas.DataFrame: A new DataFrame containing book ratings for readers in tolkien_readers.
    """
    books_of_tolkien_readers = dataset_lowercase[(dataset_lowercase['User-ID'].isin(tolkien_readers))]
    return books_of_tolkien_readers

def filter_books_to_compare(books_of_tolkien_readers, rating_threshold=8):
    """
    Extracts and returns a pandas DataFrame containing books read by readers who have read Tolkien books.

    Args:
        dataset_lowercase (pandas.DataFrame): A DataFrame containing book ratings, with columns 'User-ID',
            'ISBN', and 'Book-Rating'.
        tolkien_readers (list): A list of User-IDs who have read Tolkien books.

    Returns:
        pandas.DataFrame: A new DataFrame containing book ratings for readers in tolkien_readers.
    """
    number_of_rating_per_book = books_of_tolkien_readers.groupby(['Book-Title']).agg('count').reset_index()
    books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= rating_threshold]
    books_to_compare = books_to_compare.tolist()
    return books_to_compare

def get_top_books(books_of_tolkien_readers, books_to_compare, top_n=10):
    """
    Get the top N books from `books_to_compare` based on the mean rating given by readers who have also
    read books by Tolkien, using data from `books_of_tolkien_readers`.

    Args:
        books_of_tolkien_readers (pandas.DataFrame): A dataframe containing the ratings given by readers 
            who have also read books by Tolkien. It should have at least the columns 'User-ID', 'Book-Title', 
            and 'Book-Rating'.
        books_to_compare (list or pandas.Series): A list of book titles to compare, or a pandas Series 
            that can be used to filter `books_of_tolkien_readers` based on the 'Book-Title' column.
        top_n (int): The number of top books to return, defaults to 10.

    Returns:
        pandas.DataFrame: A dataframe containing the top N books from `books_to_compare` based on their 
        mean rating by readers who have also read books by Tolkien. The dataframe has at least the columns
        'Book-Title' and 'Book-Rating', sorted by descending rating.
    """
    filtered_books = books_of_tolkien_readers[books_of_tolkien_readers['Book-Title'].isin(books_to_compare)]
    average_ratings = filtered_books.groupby('Book-Title').agg({'Book-Rating': 'mean'}).reset_index()
    sorted_average_ratings = average_ratings.sort_values(by='Book-Rating', ascending=False)
    top_books = sorted_average_ratings.head(top_n)
    return top_books


def main():
    dataset = load_datasets()
    dataset_lowercase = filter_and_normalize_data(dataset)
    tolkien_readers = find_tolkien_readers(dataset_lowercase)
    books_of_tolkien_readers = create_books_of_tolkien_readers(dataset_lowercase, tolkien_readers)
    books_to_compare = filter_books_to_compare(books_of_tolkien_readers)
    top_books = get_top_books(books_of_tolkien_readers, books_to_compare)
    print(top_books)

if __name__ == "__main__":
    main()


