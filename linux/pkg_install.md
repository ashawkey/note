# package install 

### by package manager

```bash
sudo apt install <pkg>
```

This will install to `/usr/bin`



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





