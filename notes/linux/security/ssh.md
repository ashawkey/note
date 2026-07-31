# ssh

### ssh-gen-key & ssh-copy-id

```bash
# create a key-pair on client.
ssh-keygen

# copy it to server
ssh-copy-id user@remote_host

# private key location in server
~/.ssh/authorized_keys
```


### sshd_config

located in `/etc/ssh/sshd_config`.

```bash
# port (recommand change to larger unusual port.)
Port 22

# set root login (recommand set to no)
PermitRootLogin yes/no/prohabit-password

# whitelist / blacklist for ssh
AllowUsers user1 user2
DenyUsers user3 user4
```


