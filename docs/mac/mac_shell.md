## Mac shell

Mac use ZSH by default.

The config file is `~/.zshrc`, which needs to be created first.

```bash
# example zshrc

alias l="ls -lrth"
alias la="ls -lrtha"

alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."

alias rmr="rm -r"
alias cpr="cp -r"

alias tl="tmux ls -F '[#{session_last_attached_string}] #S'"
alias taa="tmux a"
alias ta="tmux a -t"
alias tk="tmux kill-session"
alias tn="tmux new -s"

alias proxy_on="export http_proxy=http://127.0.0.1:7897/; export https_proxy=http://127.0.0.1:7897/"
alias proxy_off="unset http_proxy; unset https_proxy"
```



### [iterm2](https://iterm2.com/)

Enable unlimited scrollback buffer:

iTerm2 --> Settings --> Profiles --> Terminal --> Check Unlimited scrooback.




### [Brew](https://brew.sh/)

The all-in-one shell package manager.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add to PATH after installation:

```bash
export PATH=/opt/homebrew/bin:$PATH
```

Usage:

```bash
# package manager
brew install git tmux htop
brew uninstall ...

# also can install apps (cask)
brew install --cask firefox

# service manager
brew services start/stop/status colima
```



### [Scroll Reverser](https://pilotmoon.com/scrollreverser/)

Configure the scroll direction of mouse & touch pad to be different.

Use brew: 

```bash
brew install scroll-reverser
```



### Docker

```bash
brew install docker docker-compose colima
colima start/stop/status
brew services start colima
docker ps
```

