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
#create_pl_csv is to create a CSV for a playlist that I am categorising its mood for (happy, sad, excited, calm)
#sentiment_score is to get the compound NLTK VADER sentiment analysis score for each song's lyrics in the playlist


#function to get all songs where the input is the playlist URI without needing the username
def create_pl_csv(playlist_uri, mood):
    mood_pl = spotify.playlist(playlist_uri)
    results = mood_pl['tracks']
    pl = results['items']
    while results['next']:
        results = spotify.next(results)
        pl.extend(results['items'])

    songs = []
    for song in pl: #[:100]: #remove comment to only get first 100 songs in the playlist
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
    df.to_csv(f'PlaylistDatasets/{mood}_playlist.csv', index=False)

    
#gets the compound sentiment analysis score of each song's lyrics in the playlist and add it to the dataframe for their respective song
def sentiment_score(mood):
    df = pd.read_csv(f'PlaylistDatasets/{mood}_playlist.csv')
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
    df['mood'] = mood #adds a column for the mood of the playlist (doing this for when I merge the CSVs later)
    df.to_csv(f'PlaylistDatasets/{mood}_playlist.csv', index=False)


#concatenating all the CSVs to create one dataset for training
def merge_csv():
    happy = pd.read_csv('PlaylistDatasets/Happy_playlist.csv')
    sad = pd.read_csv('PlaylistDatasets/Sad_playlist.csv')
    excited = pd.read_csv('PlaylistDatasets/Excited_playlist.csv')
    calm = pd.read_csv('PlaylistDatasets/Calm_playlist.csv')
    dataset = pd.concat([happy, sad, excited, calm])
    dataset.to_csv('PlaylistDatasets/Dataset.csv', index=False)

    
def remove_duplicates():
    df = pd.read_csv('PlaylistDatasets/Dataset.csv')
    df = df.drop_duplicates(['name', 'artist'], keep='first')
    df.to_csv('PlaylistDatasets/Dataset.csv', index=False)


#--- Running the functions to create CSVs for the moods ---
#create_pl_csv('spotify:playlist:4SQyVxcMYRxWfPfr5fm0JK', 'Happy')
#sentiment_score('Happy')

#create_pl_csv('spotify:playlist:0pV6pPswoQSFSDmhZ7lYeN', 'Sad')
#sentiment_score('Sad')
    
#create_pl_csv('spotify:playlist:18Phl23oHX8EtqVSv3meh2', 'Excited')
#sentiment_score('Excited')
    
#create_pl_csv('spotify:playlist:6vHM1zOGd3tRJnBHY7L5OS', 'Calm')
#sentiment_score('Calm')


#--- Merging the CSVs to create one dataset, then removes any duplicates ---
#merge_csv()
#remove_duplicates()








