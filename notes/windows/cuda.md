## CUDA on Windows


### [install](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

Very easy, just follow the [link](https://developer.nvidia.com/cuda-downloads?target_os=Windows) and get a installer.

After the installation, you can find it at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
```

Check installation by

```powershell
nvcc --version
```


### CUDA_VISIBLE_DEVICES

```powershell
set CUDA_VISIBLE_DEVICES=2,3 & python my_script.py
```


### Where is the damn `cl.exe` ?

For the error `Command '['where', 'cl']' returned non-zero exit status 1`.

  * open start menu, search `developer command prompt for VS xxxx` and start it (check the official vs code setup for [screenshots](https://code.visualstudio.com/docs/cpp/config-msvc))

  * call `where cl`, you should get something like:

    ```
    C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx86\x86\cl.exe
    ```

  * Add that directory to your PATH, and restart powershell.

  > also mentioned a script to do this  at `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat`, but I didn't find it useful...


Another brilliant solution is from `tiny-cuda-nn`, auto-find `cl` in `setup.py/backend.py`:

```python
if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++14']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path
```


  