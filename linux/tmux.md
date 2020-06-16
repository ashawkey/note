# Tmux

[--> good tutorial <--](http://louiszhai.github.io/2017/09/30/tmux/)

```bash
## session
# create session
tmux # nameless
tmux new -s session_name
# detach
tmux detach
<Ctrl + B> + d
# list sessions
tmux ls
tmux list-session
<Ctrl + B> + s
# attach
tmux a # to the first session
tmux a -t session_name
tmux attach -t session_name
# kill
tmux kill-session -t session_name
tmux kill-server # close all

## window
# create window
<Ctrl + B> + c
# switch window
<Ctrl + B> + n/p
<Ctrl + B> + 0~9
# list and select
<Ctrl + B> + W
# close window
<Ctrl + B> + &
tmux kill-window

## pane
# create
<Ctrl + B> + " # horizontal
<Ctrl + B> + % # vertical
# switch
<Ctrl + B> + o
# close
tmux kill-pane
<Ctrl + B> + x
# zoom
<Ctrl + B> + z
```

