### X.org

install: (usually you will never need to do these)

```bash
sudo apt update 
sudo apt upgrade
sudo apt install xserver-xorg-core xserver-xorg
sudo apt install xorg 
sudo apt install ubuntu-desktop
```



### ubuntu 18.04 issues

since ubuntu 18.04 changes from `lightdm` to `gdm3`, there are somethings to do if it fails to start.

* disable `wayland`:

  edit `vi /etc/gdm3/custom.conf`

  ```bash
  [deamon]
  
  WaylandEnable=false # uncomment this line
  ```

  restart gdm: `sudo systemctl restart gdm3`

  now you can see the familiar Xorg logs. (CHECK?)

* reconfigure `xorg.conf` and restart `gdm3`





### basics

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
  	ModulePath   "/usr/lib/nvidia-<xxx>/xorg" # the key line !!!
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
  
  A full example:
  
  ```bash
  Section "ServerLayout"
  	Identifier     "X.org Configured"
  	Screen         0  "Screen0" 0 0
  	InputDevice    "Mouse0" "CorePointer"
  	InputDevice    "Keyboard0" "CoreKeyboard"
  EndSection
  
  Section "Files"
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
  
  Section "Module"
  	Load  "glx"
  EndSection
  
  Section "InputDevice"
  	Identifier  "Keyboard0"
  	Driver      "kbd"
  EndSection
  
  Section "InputDevice"
  	Identifier  "Mouse0"
  	Driver      "mouse"
  	Option	    "Protocol" "auto"
  	Option	    "Device" "/dev/input/mice"
  	Option	    "ZAxisMapping" "4 5 6 7"
  EndSection
  
  Section "Monitor"
  	Identifier   "Monitor0"
  	VendorName   "Monitor Vendor"
  	ModelName    "Monitor Model"
  EndSection
  
  Section "Device"
  	Identifier  "Card0"
  	Driver      "nvidia"
  	BusID       "PCI:4:0:0"
      Option      "AllowEmptyInitialConfiguration" "true"
  EndSection
  
  Section "Screen"
  	Identifier "Screen0"
  	Device     "Card0"
  	Monitor    "Monitor0"
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     1
  	EndSubSection
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     4
  	EndSubSection
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     8
  	EndSubSection
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     15
  	EndSubSection
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     16
  	EndSubSection
  	SubSection "Display"
          Virtual 1920 1080
  		Viewport   0 0
  		Depth     24
  	EndSubSection
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





### nomachine related

```
/usr/NX/var/log/nxerror.log
```

Black screen issue for headless machines (it worked in 44!):

https://knowledgebase.nomachine.com/AR03P00973

```bash
# turn off X server
sudo systemctl stop gdm

# restart NX
sudo /etc/NX/nxserver --restart
```



### reinstall ubuntu-desktop

```bash
# do not use taskel in terminal... it stucks with no reason and once you unfocus from the tab, it somewhat freezes..


sudo apt install --reinstall ubuntu-desktop
sudo apt install --reinstall unity
```

