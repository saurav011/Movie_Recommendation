#content-based movie recommendation engine by Sougoto M.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('movie_dataset.csv')


features = ['keywords', 'cast', 'genres', 'director']
#fill the missing values with an empty string (to convert some floats to strings)
for feature in features:
    df[feature] = df[feature].fillna('')

#combining the features into a single string: Takes a row and returns the respective features of that row.
def combine_features(row):
    return row['keywords'] + ' ' + row['cast'] + ' ' + row['genres'] + ' ' + row['director']

df['combined_features'] = df.apply(combine_features, axis=1) #axis=1 combines it vertically, without it, it gets combined horizontally(columns)
df['normalized_title'] = df['title'].str.lower().str.replace(' ', '')
#df["combined_features"].head()

#counting all the features and their frequencies
vect = CountVectorizer()

matrix = vect.fit_transform(df['combined_features'])

#getting all the similarities, the movies can have.
cossim = cosine_similarity(matrix);

#getting the recommmendations of the movies which are closest to what the user likes
def get_recommendations(movie_name):
    normalized_movie_name = movie_name.lower().replace(' ', '')
    if normalized_movie_name not in df['normalized_title'].values:
        return f"No recommendations found for '{movie_name}'. Please check the movie name."
    movie_index = df[df['normalized_title'] == normalized_movie_name].index[0]
    similar_movies = list(enumerate(cossim[movie_index])) #enumerate gives the count the list of movies
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True) #reverse= True gives descending order because the more similar movies should come up first
    #the lambda funtion(minor funtion to save some lines of code) is used with the 'sorted' funtion to arrange the similar movies in a descending order iterating the whole list   
    i = 0
    number = int(input("Enter the number of suggestions you want: "))
    recommendations = []
    for movie in sorted_movies:
        movie_index = movie[0]
        recommendations.append(df.iloc[movie_index]['title'])
        i = i + 1
        if i > number:
            break
    return recommendations

#getting the input from the user of the movies he/she likes and the recommendations thereafter
movie = input("Enter the movie name: ")
recommendations = get_recommendations(movie)
print(recommendations)
