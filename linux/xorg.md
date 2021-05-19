### X.org

install:

```bash
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install xserver-xorg-core xserver-xorg
sudo apt-get install xorg 
sudo apt-get install ubuntu-desktop
```



log files:

* `/var/log/Xorg.0.log`
* `~/.xsession-error`  to see the recent errors

config:

* `/etc/X11/xorg.conf`  specifies all monitors, devices.

  to regenerate the configure file:

  ```bash
  # using X
  sudo X -configure
  
  # using nvidia
  sudo nvidia-xconfig
  ```

  [The detailed doc](https://www.x.org/releases/current/doc/man/man5/xorg.conf.5.xhtml).

  [Headless nvidia driver config](https://unix.stackexchange.com/questions/211637/how-do-i-get-x-to-start-without-a-monitor-attached-while-using-nvidia-drivers):

  ```bash
  Section "Device"
      Identifier  "Card9"
      Driver      "nvidia"
      BusID       "PCI:15:0:0"
      Option "AllowEmptyInitialConfiguration" "true" # the key line
  EndSection
  ```



* `~/.Xauthority`

  sometimes the ssh login is super slow, this maybe caused by Xauthority lock.

  solution: `sudo rm -rf ~/.Xauthority*`

* `~/.xsession`

  

  

### display manager

usually `lightdm`, alternatives are `gdm3, sddm, ...`

```bash
sudo systemctl status lightdm

# or more generally:
sudo systemctl status display-manager
```

log files:`/var/log/lightdm/lightdm.log`



### install nvidia-driver

Always use the package manager!!!

> CUDA Runfile installed version is incomplete, not fully a  graphic driver, just a GPU status monitor.
>
> For example, it will not install `nvidia-prime` which contains `prime-select`

First way is to use `ubuntu-drivers`:

```bash
sudo ubuntu-drivers devices # will list your GPU if runs normally
sudo ubuntu-drivers autoinstall # install the latest version
```

However sometimes it just output nothing, then we have to use PPA:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-418
```



### switch between intel and nvidia

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
sudo apt purge cuda*

# runfile install
# see documentation. 
```