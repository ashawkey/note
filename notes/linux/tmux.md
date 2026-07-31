# Tmux

[--> good tutorial <--](http://louiszhai.github.io/2017/09/30/tmux/)

* how to invoke tmux (remote ssh) in tmux (local) ?
  `Ctrl + b` is the outter tmux.
  `Ctrl + b + b` is the inner tmux.
  `Ctrl + b + b + b` if you have even more...

```bash
## session
# create session
tmux # nameless
tmux new -s session_name
# detach
tmux detach
<Ctrl + B> d
# list sessions
tmux ls
tmux list-session
<Ctrl + B> s
# attach
tmux a # to the first session
tmux a -t session_name
tmux attach -t session_name
# kill
tmux kill-session -t session_name
tmux kill-server # close all
# switch 
<Ctrl + B> s

## window
# create window
<Ctrl + B> c
# delete window
<Ctrl + B> d
# switch window
<Ctrl + B> 0~9
# list and select
<Ctrl + B> w

# close window
<Ctrl + B> &
tmux kill-window

## pane
# create
<Ctrl + B> " # horizontal
<Ctrl + B> % # vertical
# switch
<Ctrl + B> arrow keys
# adjust size
<Ctrl + B> (hold) + arrow keys
# close
<Ctrl + B> x

### scroll
<Ctrl + B> [
# jump to top
Alt + Shift + ,
# jump to any line in scroll mode
g 
```

