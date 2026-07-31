# cmake

### update cmake to 3.20 on ubuntu16.04

```bash
wget -c https://github.com/Kitware/CMake/releases/download/v3.20.4/cmake-3.20.4.tar.gz
tar -zxvf cmake-3.20.4.tar.gz
cd cmake-3.20.4
./bootstrap
make -j$(nproc)
sudo make install
```

