Since WSL2 has no `systemd`, we can simulate service using `nohup` by add the followings to `~/.bashrc`

```bash
# run trojan at shell launch if no trojan is running.                      
if ! pgrep "trojan" > /dev/null; then
	nohup bash ~/trojan/run.sh </dev/null >~/trojan/nohup.out 2>&1 &
fi
```



