### X.org

install:

```bash
sudo apt update 
sudo apt upgrade
sudo apt install xserver-xorg-core xserver-xorg
sudo apt install xorg 
sudo apt install ubuntu-desktop
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
  
  [Another possible case:](https://bbs.archlinux.org/viewtopic.php?id=258201)
  
  ```bash
  Section "Files"
  	ModulePath   "/usr/lib/nvidia-<xxx>/xorg" # the key line
  	ModulePath   "/usr/lib/xorg/modules"
  	FontPath     "/usr/share/fonts/X11/misc"
  	FontPath     "/usr/share/fonts/X11/cyrillic"
  	FontPath     "/usr/share/fonts/X11/100dpi/:unscaled"
  	FontPath     "/usr/share/fonts/X11/75dpi/:unscaled"
  	FontPath     "/usr/share/fonts/X11/Type1"
  	FontPath     "/usr/share/fonts/X11/100dpi"
  	FontPath     "/usr/share/fonts/X11/75dpi"
  	FontPath     "built-ins"
  EndSection
  ```
  
  It is not recommended to modify `xorg.conf` manually, but this is the only fix for my cases...



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

