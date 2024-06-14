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
```


### [iterm2](https://iterm2.com/)

Change the appearance from settings --> Profiles.


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

