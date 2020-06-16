# ffmpeg

```bash
# how does ffmpeg work?
 _______              ______________
|       |            |              |
| input |  demuxer   | encoded data |   decoder
| file  | ---------> | packets      | -----+
|_______|            |______________|      |
                                           v
                                       _________
                                      |         |
                                      | decoded |
                                      | frames  |
                                      |_________|
 ________             ______________       |
|        |           |              |      |
| output | <-------- | encoded data | <----+
| file   |   muxer   | packets      |   encoder
|________|           |______________|
```

```bash
# basics
ffmpeg -i <input> [opt] -f <format> <output>
opt:
	-an : remove audio 
	-vn : remove video
	-ss : start time, hh:mm:ss[.xxx]
	-t: continue time, 0:05
	-r: rate to extract image
	-y: overwrite without asking
	-n: never overwrite, instead exit.
	
	-loglevel: quiet, panic, ..., debug
```

```bash
# information
ffmpeg -i in.mp4
# change format
ffmpeg -i in.mp4 out.avi
# change frame rate
ffmpeg -i in.mp4 -r 24 out.mp4
# screen shot
ffmpeg -i in.mp4 -f image2 -t 0.001 -s 352x240 out.jpg # 0.001s
ffmpeg -i in.mp4 -f image2 -vf fps=fps=1/10 out%d.png # every 10s
ffmpeg -i in.mp4 out%4d.png # change all frames to image
ffmpeg -f image2 -i out%4d.png -r 25 out.mp4 # reverse, image2video

# gif
ffmpeg -i in.mp4 -vframes 30 -f gif out.gif

# extract audio
ffmpeg -i in.mp4 out.mp3

# reverse
ffmpeg -i in.mp4 -vf reverse -af reverse out.mp4
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

