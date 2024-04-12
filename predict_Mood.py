import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import lyricsgenius
import pandas as pd
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
from requests import Timeout
#nltk.download('vader_lexicon')
#nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


#defining the client id and client secret for the Spotify API
username ='21wzsn6tdh4jqkd2j75iqpz2y'
client_id = '612e708046af44a19e2d3022f877e238'
client_secret = '7d34b07c3a364b72b6f9288322570792'
redirect_uri = 'https://localhost:8888/callback'

#Genius API client access token
genius = lyricsgenius.Genius('G1bRp4mXCXyCn5G20SslMX2vWOZlVF42HXlTYVrFXm3yXVIyiZSVbxU6Dp2om7w_')
genius.timeout = 15

#client_credentials_manager is used for when I don't need specific user login
sp = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, username=username, open_browser=False)
spotify = spotipy.Spotify(client_credentials_manager=sp)


#------------------------------

def get_playlist(playlist_uri):
    pl = spotify.playlist(playlist_uri)
    results = pl['tracks']
    pl_songs = results['items']
    while results['next']:
        results = spotify.next(results)
        pl_songs.extend(results['items'])

    songs = []
    for song in pl_songs:
        song_id = song['track']['id']
        features = spotify.audio_features(song_id)[0]
        song_info = {
            "name": song['track']['name'],
            "artist": song['track']['artists'][0]['name'],
            "id": song_id,
            "danceability": features['danceability'],
            "energy": features['energy'],
            "loudness": features['loudness'],
            "speechiness": features['speechiness'],
            "acousticness": features['acousticness'],
            "instrumentalness": features['instrumentalness'],
            "liveness": features['liveness'],
            "tempo": features['tempo'],
            "valence": features['valence'],
        }
        songs.append(song_info)

    df = pd.DataFrame(songs)

    #--- NLTK VADER sentiment analysis section ---
    analyser = SentimentIntensityAnalyzer()
    sentiment_scores= []
    for song in df['name']:
        try:
            track = genius.search_song(song, df['artist'][df['name'] == song].values[0])
            if track and track.lyrics.count('\n') > 0: #if the song is found and has lyrics/more than 1 line (some songs are buggy and don't return lyrics)
                lyrics_cleaned = track.lyrics = re.sub(r"[\[].*?[\]]", "", track.lyrics) #removes text inside brackets
                lyrics_cleaned = lyrics_cleaned.split('\n', 1)[1] #removes first line of lyrics (which contains unnecessary info)
                lyrics_cleaned = lyrics_cleaned.replace('\n', ' ') #removes newline characters
                lyrics_cleaned = lyrics_cleaned.replace('  ', ' ') #removes double spaces
                sentiment = analyser.polarity_scores(lyrics_cleaned) 
                sentiment_scores.append(sentiment['compound']) #gets the compound VADER score
            else:
                sentiment_scores.append(0)
        except Timeout as e:
            print(e)
            print(f'Timeout error for {song}')
            sentiment_scores.append(0)
            continue

    df['sentiment'] = sentiment_scores
    return df

    
def training_data():
    df = pd.read_csv("PlaylistDatasets/Dataset.csv")
    df = df.drop(['id', 'liveness'], axis=1)

    x = df.iloc[:, 2:-1].values #training features of all columns (except first 3 and mood column)
    y = df.iloc[:, -1].values #target variable of mood column

    #encoding the target variable y (changes the mood column to number values)
    #le = LabelEncoder()
    #y = le.fit_transform(y)

    #splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    #standardising the data: https://simple-machine-learning-implementation-with-python.readthedocs.io/en/latest/Naive_Bayes_With_Sklearn.html
    #scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train)
    #x_test = scaler.transform(x_test)

    #prints what each encoded label means: https://stackoverflow.com/questions/42196589/any-way-to-get-mappings-of-a-label-encoder-in-python-pandas
    #le.fit(df['mood'])
    #le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    #for labelling the confusion matrix with the mood names in le_name_mapping
    #mood_names = list(le_name_mapping.keys())

    return x_train, y_train


def predict_mood(playlist_uri, pl_name):
    scaler = StandardScaler()
    x_train, y_train = training_data()
    x_train = scaler.fit_transform(x_train)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    try:
        user_playlist = get_playlist(playlist_uri)
        user_playlist = user_playlist.drop(['id', 'liveness'], axis=1)

        #this is the user's playlist as the test data
        user_test = user_playlist.iloc[:, 2:].values
        user_test = scaler.transform(user_test)

        y_pred = lr.predict(user_test)

        user_playlist['predicted_mood'] = y_pred
        user_playlist.to_csv(f'PredictedMood/{pl_name}.csv', index=False)
    except TypeError as e:
        print(e)
        print('Error with selected playlist. Please try another.')
        return None

    return user_playlist

#predict_mood('spotify:playlist:37i9dQZF1DWVRSukIED0e9', '2019')
#predict_mood('spotify:playlist:2fmTTbBkXi8pewbUvG3CeZ', '2020')
#predict_mood('spotify:playlist:5GhQiRkGuqzpWZSE7OU4Se', '2021')

# this particular playlist gives NoneType error without the try/except: predict_mood('spotify:playlist:2UPaVzzRngdrjg81VVSv7q')

