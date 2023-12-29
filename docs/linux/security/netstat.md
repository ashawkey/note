# port


### netstat

check for network communications on every port.

```bash
sudo netstat -antp # all numeric tcp program
sudo netstat -anup # ... udp ...
```


### fuser

identify processes using files or sockets.

```bash
# list user, pid, command that use the file/port.
fuser -v [-n file/udp/tcp] <file/port>

# kill them
fuser -k [-HUP/TERM/INT...] .

# interactively kill them
fuser -i -k .
```


### ufw

https://www.digitalocean.com/community/tutorials/how-to-set-up-a-firewall-with-ufw-on-ubuntu-18-04

* edit `/etc/default/ufw`

  ```
  IPV6=yes
  ```

* setup Default

  ```bash
  sudo ufw default deny incoming
  sudo ufw default allow outgoing
  
  # allow ssh
  sudo ufw allow ssh
  sudo ufw allow 22 # double check for sure
  
  # start ufw
  sudo ufw status
  sudo ufw enable # it may disrupt current ssh sessions
  
  # http
  sudo ufw allow http
  sudo ufw allow https
  
  # X11
  sudo ufw allow 6000:6007/tcp
  sudo ufw allow 6000:6007/udp
  
  # check rules & delete 
  sudo ufw status numbered
  sudo ufw delete <id>
  
  # disable 
  sudo ufw disable
  sudo ufw reset
  ```

  

