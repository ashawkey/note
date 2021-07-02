# modern commands!

A note on https://github.com/ibraheemdev/modern-unix



### `zoxide` (recommend!)

To replace `cd`

Install

```bash
curl -sS https://webinstall.dev/zoxide | bash

# add in ~/.bashrc
eval "$(zoxide init bash)"

# restart bash
```

Use

```bash
# cd <dir> but auto match
z <dir>
# interactive match
zi <dir>
```





### `mcfly`  (recommend!)

To replace `Ctrl + R`.

Install:

```bash
mkdir mcfly
cd mcfly

wget -c https://github.com/cantino/mcfly/releases/download/v0.5.6/mcfly-v0.5.6-x86_64-unknown-linux-gnu.tar.gz

tar -zxvf mcfly-v0.5.6-x86_64-unknown-linux-gnu.tar.gz
cp mcfly ~/bin/ # move to somewhere in PATH

# add to ~/.bashrc
eval "$(mcfly init bash)"
```

Use:

```bash
# initialize only once
mcfly # it takes about one minute to read in history

<Ctrl+R> # evoke it.
```





### `ag` (recommend!)

To replace `grep -rnw -e "pattern"`

Install:

```bash
sudo apt install silversearcher-ag
```

Use:

```bash
ag "pattern"
ag -i "pattern" # ignorecase
ag -w "pattern" # whole word
```





### `duf` (recommend!)

To replace `df`

Install:

```bash
sudo snap install duf-utility
```

Use:

```bash
duf
```





### `dust` (recommend!)

To replace `du`

Install:

```bash
mkdir dust
cd dust

wget -c https://github.com/bootandy/dust/releases/download/v0.6.0/dust-v0.6.0-x86_64-unknown-linux-gnu.tar.gz

tar -zxvf dust-v0.6.0-x86_64-unknown-linux-gnu.tar.gz
cd dust-v0.6.0-x86_64-unknown-linux-gnu
cp dust ~/bin/
```

Use:

```bash
dust

# limit depth == 1
dust -d 1
```





### `exa`

To replace `ls`

Install

```bash
mkdir exa
cd exa

wget -c https://github.com/ogham/exa/releases/download/v0.10.1/exa-linux-x86_64-v0.10.1.zip

unzip exa-linux-x86_64-v0.10.1.zip
cp bin/exa ~/bin/
```

Use

```bash
exa
exa -l # ls -lrtha
exa -lT # ls directory trees.

```



### `bat`

To replace `cat`

Install

```bash
mkdir bat
cd bat

wget -c https://github.com/sharkdp/bat/releases/download/v0.18.1/bat-v0.18.1-x86_64-unknown-linux-gnu.tar.gz

tar -zxvf bat-v0.18.1-x86_64-unknown-linux-gnu.tar.gz
cd bat-v0.18.1-x86_64-unknown-linux-gnu
cp bat ~/bin/
```

Use

```bash
bat <file>
```



### `gdown`

https://github.com/wkentaro/gdown

To download large files from google drive.
