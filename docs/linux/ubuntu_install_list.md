## Todo list to install ubuntu

### Rufus USB booter

https://www.ubuntu.com/download

https://rufus.ie/

Use the default settings are OK.


### Restart the machine, enter boot manager.

Zlab servers: 

* omnisky:`F11`

Select your USB as the starter, and choose `try or install ubuntu`.

To change the default boot order (e.g., if you installed new system on a different disk), you should check `Hard Disk Drive BBS Priorities`  in BIOS config.


### Installing...

Wait until you see graphics.

Choose language, how to install (alongside or erase old OS), enter your name and password.

Then another long wait...


### Setting up

* Network connection

* Basic softwares

  ```bash
  #### as the root
  apt update
  apt upgrade
  
  # necessary tools
  apt install vim git curl net-tools tmux ssh build-essential cmake htop
  
  # vision related
  apt install feh mplayer ffmpeg
  
  # if failed to fetch, install proxychains first
  sudo proxychains apt install feh
  ```

  If wish to change apt sources:

  ```bash
  cd /etc/apt
  # backup
  cp sources.list sources.list.original
  # replace the content from your source.
  vi sources.list
  ```

  TUNA source: https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

  

* VPN

  ```bash
  #### as the root
  
  ### trojan client
  wget -c https://kiui.moe/files/softwares/trojan-1.15.1-linux-amd64.tar.xz
  tar -xvf trojan-1.15.1-linux-amd64.tar.xz
  
  vi trojan/config.json
  # edit config, don't forget the password...
  
  sudo vi /etc/systemd/system/trojan.service
  # add the following
  [Unit]
  Description=trojan
  After=network.target
  
  [Service]
  Type=simple
  PIDFile=/home/kiui/trojan/trojan.pid # change the path if needed!
  ExecStart=/home/kiui/trojan/trojan -c /home/kiui/trojan/config.json -l /home/kiui/trojan/trojan.log
  ExecReload=/bin/kill -HUP \$MAINPID
  Restart=on-failure
  RestartSec=1s
  
  [Install]
  WantedBy=multi-user.target
  
  # start and test
  sudo systemctl start trojan
  sudo systemctl status trojan
  sudo systemctl enable trojan
  
  ### proxychains
  sudo apt install proxychains
  
  sudo vi /etc/proxychains.conf
  # add the followings
  socks5 127.0.0.1 1080
  
  curl -4 ip.sb
  proxychains curl -4 ip.sb
  
  ### privoxy
  sudo apt install privoxy
  
  sudo vi /etc/privoxy/config
  # add the followings
  listen-address 0.0.0.0:1081 # http proxy port
  toggle  1
  enable-remote-toggle 1
  enable-remote-http-toggle 1
  enable-edit-actions 0
  enforce-blocks 0
  buffer-limit 4096
  forwarded-connect-retries  0
  accept-intercepted-requests 0
  allow-cgi-request-crunching 0
  split-large-forms 0
  keep-alive-timeout 5
  socket-timeout 60
  
  forward-socks5 / 0.0.0.0:1080 . # trojan's socks5 proxy port
  ```

* Customization bash & tmux

  ```bash
  ### as a non-root user
  
  git clone https://github.com/ashawkey/dotfiles.git
  cd dotfiles
  bash install.sh
  
  ### set default proxy
  
  vi ~/.bashrc
  # add the followings
  export http_proxy="http://127.0.0.1:1081"
  export https_proxy="http://127.0.0.1:1081"
  
  ```

* SSH settings

  ```bash
  ### install ssh
  sudo apt install ssh # both openssh-cleint and openssh-server
  sudo systemctl enable ssh
  sudo systemctl start ssh
  
  ### ssh config
  vi /etc/ssh/sshd_config
  
  Port xxxx
  PermitRootLogin no
  AllowUsers user1 user2
  
  ### ufw
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
  sudo ufw allow 4000 # any port
  ```

* Mount Disks

  ```bash
  sudo fdisk -l 
  # should see /dev/sdb, /dev/sdc, ...
  # if partitioned, also see /dev/sdb1, /dev/sdb2, ...
  
  sudo mkdir /data2
  sudo mount /dev/sdb1 /data2
  
  # auto mount
  sudo vim /etc/fstab
  # /dev/sdb1 /data ext4 defaults 0 2
  # /dev/sdc /data2 ext4 defaults 0 2
  
  sudo mount -av
  ```

* NoMachine Remote desktop

  ```bash
  wget -c https://download.nomachine.com/download/7.10/Linux/nomachine_7.10.1_1_amd64.deb
  sudo dpkg -i nomachine_7.10.1_1_amd64.deb
  ```

  **How to support headless server**: https://kb.nomachine.com/AR03P00973

  Recommend to use the third way, although it requires manual setup at each reboot...:

  ```bash
  sudo systemctl stop gdm
  sudo /etc/NX/nxserver --restart
  
  # reconnect to nomachine, open display settings and change resolution.
  ```

* Anaconda

  ```bash
  wget -c https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  
  # if you forget to add it to path, add this in ~/.bashrc
  
  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$('/home/kiui/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/home/kiui/anaconda3/etc/profile.d/conda.sh" ]; then
          . "/home/kiui/anaconda3/etc/profile.d/conda.sh"
      else
          export PATH="/home/kiui/anaconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<
  
  # update pip
  pip install --upgrade pip
  ```

* NVIDIA driver

  Open additional drivers, choose the **latest** alternative driver for your GPU. (always use the latest, as it is usually backwards compatible to CUDA)
  
  Choose `apply changes`, and wait.
  
  Choose `restart`......
  
  Verify the driver installation by `nvidia-smi`.
  
  > No devices were found:
  >
  > This can be caused by using open-kernel driver on non-open GPU (like TITAN RTX...), see [here](https://forums.developer.nvidia.com/t/nvidia-smi-on-ubuntu-22-04-lts-no-devices-were-found/253520).
  >
  > You can find in `/var/log/syslog`:
  >
  > NVRM: Open nvidia.ko is only ready for use on Data Center GPUs.
  >
  > In such case, reinstall a driver without (open kernel) !!!
  
* CUDA

  https://developer.nvidia.com/cuda-downloads

  ```bash
  # download the correct runfile and run it to install
  # use sh, not bash, and do not run with sudo.
  sh cuda.xxxx.sh
  # just make sure the cuda version is compatible to the driver... (your driver is newer to the default one)
  # create soft link in /user/local/cuda
  
  # add to path
  export PATH="/usr/local/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
  
  nvcc -V
  ```

* CuDNN

  https://developer.nvidia.com/rdp/cudnn-download

  ```bash
  # download and uncompress
  tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
  
  # copy paste
  sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
  sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
  sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
  ```

* VS Code

  download the deb file to install.


### Misc

* I cannot open firefox or any other snap applications in nomachine virtual desktop...

  This seems to be a known bug...

  A workaround is to reinstall firefox in dev mode:

  ```bash
  sudo snap remove firefox
  sudo snap install firefox --devmode
  
  # same for the other apps
  sudo snap remove snap-store
  sudo snap install snap-store --devmode
  ```

  

