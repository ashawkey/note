# Qt

Qt is a fucking commercial software, and all its free releases are to disgust you.



You must register to download even the free version.

Docs at: https://doc.qt.io/qt-5/gettingstarted.html

TUNA mirror (always god!): https://mirrors.tuna.tsinghua.edu.cn/qt/

Official (slow): https://www.qt.io/offline-installers 



To manage different Qt versions, use `PATH` and `LD_LIBRARY_PATH`.



### Qt 5.5.1 ~ 5.9.7

The default version, in `/usr/libx86_64-linux-gnu/`. 

what `sudo apt install qt-default` will install. 

However, this is terribly outdated, e.g., `PyQt5, PySide2` do not support it.



### Qt 5.12.11

The last version that supports a GUI installer.

download at: https://mirrors.tuna.tsinghua.edu.cn/qt/official_releases/qt/5.12/5.12.11/

```bash
wget -c https://mirrors.tuna.tsinghua.edu.cn/qt/official_releases/qt/5.12/5.12.11/qt-opensource-linux-x64-5.12.11.run

chmod u+x ./qt-opensource-linux-x64-5.12.11.run
sudo ./qt-opensource-linux-x64-5.12.11.run
```



Install the following `[Yes]`, which takes ~ 1.5G:

```
-- Qt 5.15.11
    -- gcc_64 [Yes]
    -- Sources [No, don't need] 
    -- Qt Charts [Yes]
    -- Qt PDF [Yes]
    -- Qt ... [Yes]
    -- Qt Script (Deprecated) [Yes]
-- Qt Creator [Yes]
```

It is installed in `/opt/Qt5.12.11/` by default, and you need to manually add to `~/.bashrc`:

```bash
export PATH="/opt/Qt5.12.11/5.12.11/gcc_64/bin:$PATH"
export LD_LIBRARY_PATH="/opt/Qt5.12.11/5.12.11/gcc_64/lib:$LD_LIBRARY_PATH"
```

check by:

```bash
. ~/.bashrc
qmake -v
# should output Qt5.12.11 at /opt/Qt5.12.11/5.12.11/lib
```



##### PyQt5 support

```bash
pip install PyQt5==5.12.11
# do not install the default version, which is for 5.15
```





### Qt 5.15

You must build it from source. 

download at: https://mirrors.tuna.tsinghua.edu.cn/qt/official_releases/qt/5.15/5.15.2/single/

docs at: https://doc.qt.io/qt-5/build-sources.html



