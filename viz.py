import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  ratings DF
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='cp1251', sep=';',low_memory=False)
ratings = ratings[ratings['Book-Rating'] != 0]

#  books DF
books = pd.read_csv('BX-Books.csv', encoding='cp1251', sep=';', error_bad_lines=False,low_memory=False)

# Outliers box plot
plt.figure(figsize=(10, 6))
plt.boxplot(ratings['Book-Rating'], vert=False)
plt.xlabel('Book-Rating')
plt.title('Boxplot of Book Ratings')
plt.grid(axis='x', alpha=0.75)
plt.show()

