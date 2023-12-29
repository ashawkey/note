# Magic SysRq


`SysRq` is usually `Print Screen` in keyboard.


- `Ctrl+Z`: pause a process (plus `bg` to resume in the background, `fg` to raise to foreground)
- `Ctrl+C`: politely ask the process to shut down now
- `Ctrl+\`: mercilessly kill the process that is currently in the foreground


Push each key sequentially.

- `Alt+SysRq+s`: Write data to disk (always do this before killing anything important)
- `Alt+SysRq+s, k`: mercilessly kill all current processes on a given virtual console
- `Alt+SysRq+s, b`: mercilessly reboot without unmounting,
- `Alt+SysRq+r, e, i, s, u, b`: Safely **r**eboot **e**ven **i**f the **s**ystem is **u**tterly **b**roken

