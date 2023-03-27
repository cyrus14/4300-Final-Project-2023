# https://spotipy.readthedocs.io/en/2.22.1/
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import json

CREDS = json.load(open('./conf.json', 'r+'))

# Lol i dont care rn 
CLIENT_ID = CREDS['spotify_client']
CLIENT_PRIVATE = CREDS['spotify_private']

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_PRIVATE))

artist = "Miles%20Davis"
track = "Doxy"


print(
    sp.search(q="track\:nobody%20artist\:mitski", limit=5, type='track')
)