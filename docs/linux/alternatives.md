# update-alternatives


### gcc example

```bash
# install gcc/g++ 7
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y

# clear 
sudo update-alternatives --remove-all gcc 
sudo update-alternatives --remove-all g++

# install (50/60 is the priority for auto mode)
# slave g++ to gcc, so we only need to config gcc once.
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-5 

# config
sudo update-alternatives --config gcc
```

