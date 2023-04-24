import os
import requests
import pandas as pd

# Function to download files
def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# Download the data files
ratings_link = "https://link"
books_link = "https://link"

ratings_file = "ratings.csv"
books_file = "books.csv"

if not os.path.exists(ratings_file):
    download_file(ratings_link, ratings_file)

if not os.path.exists(books_file):
    download_file(books_link, books_file)

# Load ratings
ratings = pd.read_csv(ratings_file)
ratings = ratings[ratings['rating'] != 0]