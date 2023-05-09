# Book Recommender


![Pandas](https://img.shields.io/badge/pandas-1.1.5-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-1.19.5-blue.svg)
![FuzzyWuzzy](https://img.shields.io/badge/fuzzywuzzy-0.18.0-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-0.84.0-blue.svg)

Book Recommender - A web app that utilizes pandas, NumPy, FuzzyWuzzy, and Streamlit for UI to recommend books based on their correlations with a user-selected book, using a testing dataset. This test assignment script is built upon the proof of concept for a data scientist's algorithm.

## Features

- Get top 10 book recommendations for a selected book
- Filter recommendations based on a minimum number of common ratings
- Search for books with FuzzyWuzzy's string matching algorithm

## Installation

1. Clone the repository:

```
https://github.com/sebi83/A-book-recommender-based-on-test-dataset.git
```

2. Change into the project directory

```

cd <path to cloned repository </path> >

```


3. Install the required packages:

```
pip install -r requirements.txt
```

4. Run the Streamlit application:

```
streamlit run app.py
```

## Usage

1. Enter a book title in the search bar.
2. Select a book from the list of matching titles.
3. View the top 10 book recommendations, sorted by correlation and average rating.


## Helpers

In repo there is a viz.py boxplot helper for detecting outliers, also a custom df_downloader.py which has to have the same structure as a CSV test file and can customize provided code though its a only a assignment and can be done through streamlit upload features.


## License

MIT License. See [LICENSE](LICENSE) for more information.
