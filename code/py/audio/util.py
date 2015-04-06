__author__ = 'jeffreyquinn'
from collections import namedtuple

Metadata = namedtuple('Metadata',
                      ('title',       # the title of the track
                       'artist',      # the artist of the track
                       'album',       # the album on hich this track appears
                       'comment',     # the comment field stored in the tags
                       'genre',       # the genre as stored in the tags
                       'track',       # the track number
                       'year',        # the year of publication
                       'length',      # the length of the track, in seconds
                       'bitrate',     # the bitrate of the track [kb/s]
                       'sampleRate',  # the sample rate
                       'channels'     # the number of channels
                     ))

