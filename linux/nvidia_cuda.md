# Nvidia driver and CUDA

CUDA and nvidia-driver can be safely separately installed. 

Usually, we install nvidia-driver first since it is necessary for GUI.

CUDA is only used for high-performance computation.



### install nvidia-driver

Always use the package manager!!!

> CUDA Runfile installed version is incomplete, not fully a  graphic driver, just a GPU status monitor.
>
> For example, it will not install `nvidia-prime` which contains `prime-select`

First, go runlevel 2 and turn off any GUI.

```bash
# go runlevel 3
sudo systemctl isolate multi-user.target # will stop all the graphic processes
sudo telinit 3
```

Recommended way is to use `ubuntu-drivers`:

```bash
# will list your GPU and suitable driver versions
sudo ubuntu-drivers devices 
# install the latest version
sudo ubuntu-drivers autoinstall 
```

However sometimes it just output nothing, then we have to use PPA:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
# e.g. choose a suitable version manually.
sudo apt install nvidia-418
```

Finally, go runlevel 5 and reboot.

```bash
# go runlevel 5
sudo systemctl isolate graphical.target
sudo telinit 5

# set default runlevel 5
sudo systemctl enable graphical.target
sudo systemctl set-default graphical.target

sudo reboot
```

Location of `nvidia-smi`:

```bash
nvidia-debugdump -> /etc/alternatives/x86_64-linux-gnu_nvidia-debugdump
nvidia-xconfig -> /etc/alternatives/x86_64-linux-gnu_nvidia_xconfig
nvidia-smi -> /etc/alternatives/x86_64-linux-gnu_nvidia_smi
.
```



### switch between intel and nvidia driver

```bash
# in case prime-select is not found
sudo apt install nvidia-prime

# show current video driver
sudo prime-select query

# change
sudo prime-select nvidia
sudo prime-select intel
```



### uninstall nvidia-driver

```bash
# pkg manager install
sudo apt purge nvidia*
```



### install CUDA

Recommend to use Runfile install. 

* We can install any version we want, and handle multiple versions by setting PATH and LD_LIBRARY_PATH.
* We can easily uninstall it. (just rm it)
* Package manager may need VPN to download CUDA.

Download here: https://developer.nvidia.com/cuda-downloads

AND The detailed installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Do not need to go runlevel 3 or turn off GUI to install CUDA.

```bash
# install
# do not check the driver install !!! we have installed it separately.
sudo bash cuda_<version>_linux.run

# check installed version
nvcc -V
```



### switch between multiple CUDA versions

Globally: change the `/usr/local/cuda` soft link.

```bash
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
```

Locally: change your `PATH` and `LD_LIBRARY_PATH`:

```bash
export PATH="/usr/local/cuda-9.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.1/lib64:$:LD_LIBRARY_PATH"

export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$:LD_LIBRARY_PATH"
```



### uninstall CUDA

Just REMOVE it.

```bash
sudo rm -rf /usr/local/cuda-10.1
```

If you use package manager to install CUDA:

```bash
sudo apt purge cuda*
```



### pitfalls

* for CUDA 10.1, install the update `10.1.243` directly, instead of the general release `10.1.105` !!! Or you can update from `105` to `243`, just run the new `runfile` and it will prompt.

  