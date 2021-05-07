### X.org

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





### switch between intel and nvidia

```bash
# show current video driver
sudo prime-select query

# change
sudo prime-select nvidia
sudo prime-select intel
```



### uninstall nvidia-driver

```bash
# pkm install
sudo apt purge nvidia*
sudo apt purge cuda*

# runfile install
# see documentation. 
```