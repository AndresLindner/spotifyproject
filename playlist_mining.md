Playlist Mining

For this project we centered our data on existing playlists from spotify. We decided to get these playlists from two different curators. Spotify’s public playlists which are a combination of machine learning generated playlists and professional music curators  and the EveryNoise.com playlist set which is a compilation of playlists for around 1500 genres. EveryNoise.com uses acoustic similarity to generate these playlists. We found that some of these playlists were actually very good when looking at the playlist generated for our favorites musical genres. We decided that we could leverage on some of the already generated playlists to train from them using different features.

API

Using the spotify api we were able to get a very great range of features. We realized that spotify has done a very good job on incorporating new features to their api. Audio features data used for music similarity is now also available for spotify. Playlist enhancement features like tags was done used the musicbrainz database. 

MSE
We decided to use average and standard deviation for each valuer where we believe that we might lose data diversity when doing averages

Engineered Features

Playlist Name Score ( discoverability is a playlist)

One of the things that surprised us the lack of exploratory functions in the spotify api. It’s a database with around 1 billion playlists however the main exploratory function is the search api, which lets you only scratch the very top of the richness of this database since it only lets you search by words. These words have to be associated with the playlist name. In order for a playlist to be found it has to be related to a keyword in the search bar. This gave us a hint, depending how easy is to find a playlist it’s more likely that the playlist have more followers but how can we determine this?. We decided to mine the musicbrainz database to obtain keyword tags associated with the artist or songs in the playlists. We engineered a playlist parameter based on the frequency of the words associated with the playlists and its appearance the playlist name. We performed searches for random tags associated with artist like jay-z; Words like ‘new york’, ‘grammy winner’ and ‘brooklyn rap’ include playlists that includes this artist.

namescore =  x/n 
where:
 x is the frequency of a word in the playlist name was associated with an artist
 n is the number of word associated with all the artists in the playlist





Top Genere Ratio

We want to explore how the ratio of of the most popular genre in the playlist effects related to the number of followers in a playlist. We came with this idea from some blogs that suggest that successful playlists would a minimum ratio of songs for a specific playlist
ratio =  x/n 
where:
 x is the frequency of an artist genre in the playlist
 n is the number artist genres in the playlist

Decades Ratio

We want to explore how the ratio of the age of the tracks in a playlists affect the prediction. This ratio would allow us to understand how diversified in term of old/new music is included in the same playlist and how this diversification affects the number of followers

ratio =  x/n 
where:
 x is the number of tracks released during a decade in the playlist
 n is the number of tracks the playlist
