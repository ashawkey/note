## ffmpeg quick manual

```bash
### basics
ffmpeg -i in.avi out.mp4
ffmpeg -y -i in.avi out.mp4 # overwrite

### cut 
# format is hh:mm:ss
ffmpeg -i in.mp4 -ss 00:00:05 -to 00:05:15 out.mp4

### crop
ffmpeg -i in.mp4 -vf "crop=out_w:out_h:x:y" out.mp4

### rescale
ffmpeg -i in.mp4 -vf "scale=out_w:out_h" out.mp4

# change FPS
ffmpeg -i in.mp4 -vf "fps=fps=30" out.mp4
ffmpeg -i in.mp4 -r 30 out.mp4

# merge audio to video
ffmpeg -i in.mp4 -i aud.wav -c:v copy -c:a aac out.mp4

# extract audio from video
ffmpeg -i in.mp4 -f wav -ar 16000 out.wav

# extract image frames from video
ffmpeg -i in.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 "%d.jpg"
```

