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

sudo ufw allow ssh # by default it opens 22 port
sudo ufw allow http
sudo ufw allow https
sudo ufw allow ‘Nginx Full’
sudo ufw allow 20212 # any port

# see allow list
sudo ufw app list

# enable 
sudo ufw enable
```

