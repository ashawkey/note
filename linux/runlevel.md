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
```



### change current runlevel

```bash
sudo systemctl isolate multi-user.target
# will stop all the graphic processes
```



### set default runlevel

```bash
sudo systemctl enable multi-user.target
sudo systemctl set-default multi-user.target
sudo reboot
```

