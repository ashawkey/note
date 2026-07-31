### ffmpeg

```bash
### info
ffmpeg -i in.mp4

### change format
ffmpeg -i in.avi out.mp4
ffmpeg -y -i in.avi out.mp4 # overwrite

### cut 
# format is hh:mm:ss
ffmpeg -i in.mp4 -ss 00:00:05 -to 00:05:15 out.mp4

### crop
ffmpeg -i in.mp4 -vf "crop=out_w:out_h:x:y" out.mp4

### rescale
ffmpeg -i in.mp4 -vf "scale=out_w:out_h" out.mp4

### change FPS
ffmpeg -i in.mp4 -vf "fps=fps=30" out.mp4
ffmpeg -i in.mp4 -r 30 out.mp4

### change duration
# e.g., in.mp4 is 46s, you want it to speed up and finish in 30s
# calculate target/current ratio 30/46=0.6521
ffmpeg -i in.mp4 -vf "setpts=0.6521*PTS" out.mp4

### merge audio to video
ffmpeg -i in.mp4 -i aud.wav -c:v copy -c:a aac out.mp4

### extract audio from video
ffmpeg -i in.mp4 -f wav -ar 16000 out.wav

### extract image frames from video
ffmpeg -i in.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 "%d.jpg"

### concatenate image to video
# for 0000, 0001, ... name convention
ffmpeg -framerate 30 -i %04d.png -vcodec libx264 -pix_fmt yuv420p out.mp4
ffmpeg -framerate 10 -pattern_type glob -i "*.jpg" -c:v libx264 -pix_fmt yuv420p out.mp4
# for 0, 1, 2, ..., 10 name convention (without leading 0 paddings)
cat $(find . -name '*.jpg' -print | sort -V) | ffmpeg -framerate 10 -i - -c:v libx264 -pix_fmt yuv420p out.mp4

### compress visually-losslessly
# crf: compression levvel, high values lead to smaller file with worse quality.
ffmpeg -i in.mp4 -c:v libx264 -crf 18 -preset veryslow -c:a copy out.mp4

### concat videos sequentially
# create a file `list.txt` containing the videos you want to concat
file a.mp4
file b.mp4
# run
ffmpeg -safe 0 -f concat -i list.txt -c copy out.mp4
```





## ffplay

```bash
ffplay [opt] <input>
-x, -y : force h,w
-fs : full screen
-sn : disable subtitles
-an, -vn : disable audio/vidio
-ss : start position
-t : duration
-loop <num> : 0 means forever
-showmode <mode> : 0:video, 1:waves, 2:rdft

### while playing
q, ESC
f # fs
p, SPC # pause
m # mute
/, * # volume

```

