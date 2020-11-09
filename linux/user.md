# User

* Add user

  ```bash
  # do all the defaults: create home, select shell, prompt for password.
  adduser hawkey
  
  # do nothing
  useradd hawkey
  useradd -d /home/hawkey -s /bin/bash 
  
  passwd hawkey # change password
  ```

  

* Delete user

  ```bash
  userdel hawkey
  userdel -r hawkey # delete home
  ```

  

* switch user

  ```bash
  su hawkey
  su # root
  ```

  

* add sudoer

  ```bash
  usermod -aG sudo hawkey
  ```

  Or

  ```bash
  visudo # however, this is nano
  vim /etc/sudoers
  ```

  and append a line:

  ```
  hawkey ALL=(ALL) NOPASSWD:ALL
  ```

  

* Change shell

  ```bash
  chsh -s /bin/bash # but never change shell for root !!!
  ```

  Or 

  ```bash
  vi /etc/passwd
  ```

  and modify the corresponding line.