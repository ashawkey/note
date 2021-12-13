# package install 

### apt package manager

```bash
sudo apt install <pkg>
```

This will install to `/usr/bin`

```bash
sudo apt remove <pkg> # just remove the binary, if re-installed, your configurations will be restored.
sudo apt purge <pkg> # also clear everything, e.g., configuration files

sudo apt show <pkg> # show information

sudo apt list --installed
sudo apt list --upgradable

apt search <pkgname>
# e.g. apt search ^libxxx*
```

apt will read & download sources from `/etc/apt/sources.list`

We recommend to use the [TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/).





### build from source

```bash
wget <pkg_source.tar.gz>
tar -zxvf <pkg_source.tar.gz>
cd <pkg_source>
mkdir build
cd build
cmake ..
make
sudo make install
```

This will install to `/usr/local/bin`



### PPA

Personal Package Achieve. To install **unofficial** softwires by `apt`.

```bash
sudo add-apt-repository ppa:dr-akulavich/lighttable
sudo apt update
sudo apt install lighttable-installer
```

This will add the PPA into `/etc/apt/sources.list.d`, which is included into `/etc/apt/sources.list`  for `apt update` to find it.

To remove PPA and installed packages, we need `synaptic`:

```bash
sudo apt install synaptic
```

Then we can use the GUI to remove them.

Also, we can just remove them in CLI:

```bash
sudo rm /etc/atp/sources.list.d/xxx.list*
```





Remove trusted GPG keys:

```bash
# list keys
$ sudo apt-key list
/etc/apt/trusted.gpg
--------------------
pub   1024D/437D05B5 2004-09-12
uid                  Ubuntu Archive Automatic Signing Key <ftpmaster@ubuntu.com>
sub   2048g/79164387 2004-09-12

pub   4096R/C0B21F32 2012-05-11
uid                  Ubuntu Archive Automatic Signing Key (2012) <ftpmaster@ubuntu.com>

pub   4096R/EFE21092 2012-05-11
uid                  Ubuntu CD Image Automatic Signing Key (2012) <cdimage@ubuntu.com>

pub   1024D/FBB75451 2004-12-30
uid                  Ubuntu CD Image Automatic Signing Key <cdimage@ubuntu.com>

pub   2048R/BE1229CF 2015-10-28
uid                  Microsoft (Release signing) <gpgsecurity@microsoft.com>

/etc/apt/trusted.gpg.d/graphics-drivers_ubuntu_ppa.gpg
------------------------------------------------------
pub   4096R/1118213C 2015-08-12
uid                  Launchpad PPA for Graphics Drivers Team

/etc/apt/trusted.gpg.d/microsoft.gpg
------------------------------------
pub   2048R/BE1229CF 2015-10-28
uid                  Microsoft (Release signing) <gpgsecurity@microsoft.com>

# to delete 
$ sudo apt-key del BE1229CF
```



### unmet dependencies

Sometimes we meet errors like:

```bash
$ sudo apt-get install xorg
Reading package lists... Done
Building dependency tree
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies:
 xorg : Depends: xserver-xorg (>= 1:7.7+13ubuntu3)
        Depends: libgl1-mesa-glx but it is not going to be installed or
                 libgl1
        Depends: libglu1-mesa but it is not going to be installed
        Depends: x11-utils
E: Unable to correct problems, you have held broken packages.
```

This is mainly caused by a package is installed from PPA, and we should **downgrade** it.

We can use `aptitude` for further information.

```bash
$ sudo apt install aptitiude
$ sudo aptitude install xserver-xorg-core
The following NEW packages will be installed:
  libgl1-mesa-glx{ab} xserver-xorg-core
0 packages upgraded, 2 newly installed, 0 to remove and 0 not upgraded.
Need to get 1,467 kB of archives. After unpacking 4,545 kB will be used.
The following packages have unmet dependencies:
 libgl1-mesa-glx : Depends: libglapi-mesa (= 11.2.0-1ubuntu2) but 17.2.8-0ubuntu0~16.04.1 is installed.
The following actions will resolve these dependencies:

     Keep the following packages at their current version:
1)     libgl1-mesa-glx [Not Installed]
2)     xserver-xorg-core [Not Installed]
```

Highlight: **Depends: libglapi-mesa (= 11.2.0-1ubuntu2) but 17.2.8-0ubuntu0~16.04.1 is installed.** 

We can manually downgrade by:

```bash
sudo apt install libglapi-mesa=11.2.0-1ubuntu2
sudo apt install xorg # ok!
```

Or, `aptitude` will suggest you how to repair, just type `n` to select the preferred solution and type `Y`.





