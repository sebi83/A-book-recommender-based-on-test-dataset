import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load ratings
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';')
ratings = ratings[ratings['Book-Rating'] != 0]

# Load books
books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';', error_bad_lines=False)

# Plot the ratings distribution
plt.figure(figsize=(10, 6))
plt.boxplot(ratings['Book-Rating'], vert=False)
plt.xlabel('Book-Rating')
plt.title('Boxplot of Book Ratings')
plt.grid(axis='x', alpha=0.75)
plt.show()
ls
