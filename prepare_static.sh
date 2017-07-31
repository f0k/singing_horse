#!/bin/bash
# Pass some audio file as the first argument.
# Will take 0:00:01.5 -- 0:00:08.5 as the 7-second example snippet.
here="${0%/*}"
background_audio="$1"
ffmpeg -y -i "$background_audio" -ss 1.5 -t 7 "$here"/static/original.mp3
python "$here"/spects.py "$here"/static/original.{mp3,npy}
python "$here"/spects.py "$here"/static/original.{npy,png}
python "$here"/spects.py "$here"/static/{original.npy,melspect.mp3}
optipng -p -q "$here"/static/original.png
