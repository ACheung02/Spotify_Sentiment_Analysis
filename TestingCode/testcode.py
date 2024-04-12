import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import lyricsgenius
import pandas as pd
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
#nltk.download('vader_lexicon')
#nltk.download('stopwords')

#defining the client id and client secret for the Spotify API
username ='21wzsn6tdh4jqkd2j75iqpz2y'
client_id = '612e708046af44a19e2d3022f877e238'
client_secret = '7d34b07c3a364b72b6f9288322570792'
redirect_uri = 'https://localhost:8888/callback'
#scope = "user-library-read" #if needed, add scope=scope to the SpotifyOAuth function

#Genius API client access token
genius = lyricsgenius.Genius('G1bRp4mXCXyCn5G20SslMX2vWOZlVF42HXlTYVrFXm3yXVIyiZSVbxU6Dp2om7w_')
genius.timeout = 15

#client_credentials_manager is used for when I don't need specific user login
sp = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, username=username, open_browser=False)
spotify = spotipy.Spotify(client_credentials_manager=sp)

#----- experimenting with Genius API and lyricsGenius -----

#gets all songs from a playlist 
# pl2020 = spotify.playlist('spotify:playlist:2fmTTbBkXi8pewbUvG3CeZ')
# results = pl2020['tracks']
# pl = results['items']
# while results['next']:
#     results = spotify.next(results)
#     pl.extend(results['items'])
              
# prints the name and artist of each song in the playlist              
# for song in pl:
#     print(song['track']['name'], "by", song['track']['artists'][0]['name'])


#prints the audio features of the first song in pl
# song_id = pl[0]['track']['id']
# features = spotify.audio_features(song_id)[0]
# print(features)


#gets the song url of the first song in pl
# song_url = pl[0]['track']['external_urls']['spotify']
# print(song_url)


#searches for the lyrics of song_url
# song = genius.search_song(pl[0]['track']['name'], pl[0]['track']['artists'][0]['name'])
# print(song.lyrics)

    
# -----------------------------------------------------------------------------------------

#creating a dataframe of the playlist
# songs = []
# for song in pl:
#     song_id = song['track']['id']
#     features = spotify.audio_features(song_id)[0]
#     song_info = {
#         "name": song['track']['name'],
#         "artist": song['track']['artists'][0]['name'],
#         "id": song_id,
#         "danceability": features['danceability'],
#         "energy": features['energy'],
#         "loudness": features['loudness'],
#         "speechiness": features['speechiness'],
#         "acousticness": features['acousticness'],
#         "instrumentalness": features['instrumentalness'],
#         "liveness": features['liveness'],
#         "valence": features['valence'],
#         "tempo": features['tempo'],
#     }
#     songs.append(song_info)
    
# df = pd.DataFrame(songs)
# df.to_csv('pl2020.csv', index=False)


# ----- practice with nltk vader sentiment analysis -----

analyser = SentimentIntensityAnalyzer()

# finding the sentiment score of the lyrics
#track = genius.search_song(pl[0]['track']['name'], pl[0]['track']['artists'][0]['name']) #gets first song in the playlist

#https://stackoverflow.com/questions/65022050/cleaning-song-lyrics-with-regex used this to remove text inside brackets
track = genius.search_song("My Kind of Woman", "Mac DeMarco")
lyrics_cleaned = track.lyrics = re.sub(r"[\[].*?[\]]", "", track.lyrics) #remove text inside brackets
lyrics_cleaned = lyrics_cleaned.split('\n', 1)[1] #remove first line of lyrics (which contains unnecessary info)
lyrics_cleaned = lyrics_cleaned.replace('\n', ' ') #remove newline characters
lyrics_cleaned = lyrics_cleaned.replace('  ', ' ') #remove double spaces

# --- removes stop words (although VADER already does this so not necessary in final code)
#stop_words = stopwords.words('english')
#lyrics_cleaned = ' '.join([word for word in lyrics_cleaned.split() if word not in stop_words]) 

sentiment_score = analyser.polarity_scores(lyrics_cleaned)
print(lyrics_cleaned)
print('Score:', sentiment_score)


#finds the sentiment score of each song in the playlist
# sentiment_scores = []
# for song in pl:
#     track = genius.search_song(song['track']['name'], song['track']['artists'][0]['name'])
#     sentiment_score = analyser.polarity_scores(track.lyrics)
#     sentiment_scores.append(sentiment_score)

#adds only the compound sentiment score to the dataframe for each respective song in the playlist
# compound_scores = []
# for score in sentiment_scores:
#     compound_scores.append(score['compound'])   
# df['sentiment'] = compound_scores
# df.to_csv('pl2020.csv', index=False)













