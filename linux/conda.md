# conda



### envs

```bash
# list all
conda env list

# create
conda env create -n <name> python=3.7

# delete
conda env remove -n <name>
```



### conda file locations

In `.bashrc`:

```bash
# auto matically added.
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/tang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/tang/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/tang/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/tang/anaconda3/bin:$PATH"
    fi  
fi
unset __conda_setup
# <<< conda initialize <<<

# if you want to use the anaconda libs instead of global libs:
export LD_LIBRARY_PATH="/home/tang/anaconda3/lib:${LD_LIBRARY_PATH}"
```

The directory structures:

```bash
### the installed binaries and libraries
# global
~/anaconda3/bin
~/anaconda3/lib
# for each env
~/anaconda3/envs/<name>/lib
~/anaconda3/envs/<name>/bin

### the installed python packages for each env
~/anaconda3/envs/<name>/lib/python3.7/site-packages
```



### package management

```bash
# install
conda install <pkg> # from conda source.
pip install <pkg> # from pip source, but also install to the current env

# list 
conda list

# update
conda update <pkg>

# uninstall
conda uninstall <pkg>

# search
conda search <pkg>

# export to / install from a text file
conda list --explicit > env.txt
conda env create --file env.txt
```

