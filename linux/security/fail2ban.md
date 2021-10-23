# fail2ban



```bash
### install
sudo apt install fail2ban

### check status
sudo systemctl status fail2ban

### config
sudo cp /etc/fail2ban/jail.{conf,local} # create a local config copy
sudo vim /etc/fail2ban/jail.local

# bantime = 1d
# findtime = 10m (all entries should be in this time range to be counted.)
# maxentry = 5


### check all banned ips
# by logs
sudo zgrep 'Ban' /var/log/fail2ban.log*
# or directly by iptables
sudo iptables -L -n | awk '$1=="REJECT" && $4!="0.0.0.0/0"'
```

