# opencv

### install opencv2

Although opencv2 has been outdated, lots of packages still use it...

However, do not compile with CUDA >=10.0. It has severe bugs.

```bash
### Download opencv-2.4.13.5
wget https://github.com/opencv/opencv/archive/2.4.13.5.zip -O opencv-2.4.13.5.zip
unzip opencv-2.4.13.5.zip
cd opencv-2.4.13.5
mkdir release
cd release

### Compile and install
cmake -D WITH_CUDA=OFF .. # better turn off for cuda > 10.0
make all -j$(nproc) # Uses all machine cores
sudo make install

### Echoes OpenCV installed version if installation process was successful
echo -e "OpenCV version:"
pkg-config --modversion opencv
```

