# ufw


### enable

```bash
sudo apt install ufw

# enable ipv6
sudo vim /etc/default/ufw
# set `IPV6=yes`

# set rules
sudo ufw default deny incoming
sudo ufw default allow outgoing

# allow
sudo ufw allow ssh # by default it opens 22 port
sudo ufw allow http
sudo ufw allow https
sudo ufw allow ‘Nginx Full’
sudo ufw allow 20212 # any port

# deny
sudo ufw deny 20212

# see rule list
sudo ufw status [verbose/numbered]

# delete rules by id
sudo ufw status numbered # check the id
sudo ufw delete <id>

# or delete by name
sudo ufw delete allow 20212
sudo ufw delete deny 20212

# enable 
sudo ufw enable
```

