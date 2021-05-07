 # runlevel

```
 ┌─────────┬───────────────────┐
 │Runlevel │ Target            │
 ├─────────┼───────────────────┤
 │0        │ poweroff.target   │
 ├─────────┼───────────────────┤
 │1        │ rescue.target     │
 ├─────────┼───────────────────┤
 │2, 3, 4  │ multi-user.target │
 ├─────────┼───────────────────┤
 │5        │ graphical.target  │
 ├─────────┼───────────────────┤
 │6        │ reboot.target     │
 └─────────┴───────────────────┘
```



### check runlevel

```bash
# show previous and current runlevel
runlevel 
# e.g. `N 5`, `5 3`

sudo systemctl get-default
```



### change current runlevel

```bash
# go runlevel 3
sudo systemctl isolate multi-user.target # will stop all the graphic processes
sudo telinit 3

# go runlevel 5
sudo systemctl isolate graphical.target
sudo telinit 5
```



### set default runlevel

```bash
# go runlevel 3
sudo systemctl enable multi-user.target
sudo systemctl set-default multi-user.target

# go runlevel 5
sudo systemctl enable graphical.target
sudo systemctl set-default graphical.target

# don't forget to reboot!
sudo reboot
```



However, if lightdm failed to start, your runlevel will still be 3 even if the default is 5.

check lightdm status and xorg log.