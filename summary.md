---
title: Project Trajectory and Summary
nav_include: 6
---

### Approach and Project Trajectory

1.
Then we built more complex models using random forest regression and gradient boosted regression trees and were
able to improve the test R-squared to about 63%

2.
Next we added features from the million song database to provide more data on the artists and the decade to which
the song belongs. These additional features did not help improve the model

3.
We then added features related to the artist genres - which is a statistic about the number of genres represented.
We got the genre info from a website called 'everyNoise'.

4.
We engineered a playlist parameter based on the frequency of the words associated with the playlists and its
appearance the playlist name.


5. Based on the above engineered features we were able to improve the test R-Squared value to about 75%


### Summary
We can predict the success of a playlist based on the followeing features a)How recently is the playlist updated b)How long has the playlist been active 
c) Popularity of the song d)Popularity of the artists d) Number of songs in the playlist 

In addition acoustic features such as speechiness and danceability also have an impact on how popular the playlist gets


### Future work
1. Given time, we would like to add more features related to the audio data analysis
2. We want to consider more databases such as wikipedia and look for possible features
3. Given time, we would like generate playlists based on more features related to the audio analysis than we currently considered
