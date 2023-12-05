## powershell

### set execution policy
The default is `restricted`, which disables executing custom ps scripts (ps1 files).
Run powershell in administrator mode (ctrl + click new terminal), then:
```powershell
set-executionpolicy remotesigned

# check
get-executionpolicy # show remotesigned
```


### profile (bashrc)

```powershell
# location
$profile
# C:\Users\haw\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1
# if it doesn't exist, you need to create it manually...

# edit it
notepad $profile
```

an example:

```powershell
set-alias l ls
set-alias npp notepad
set-alias grep select-string

function lN
{
    ls -Name
}
function la
{
    ls -hidden
}

function ..
{
    cd ..
}

function W
{
    set-location E:\aa
}

# recursively find files, mimicking find.
function find([string] $glob)
{
    ls -recurse -include $glob
}

# path
#$env:Path += ";C:\Program Files\Racket"
```

reload:

```powershell
. $profile
```

### set proxy via profile
Add in your `$profile`:
```powershell
# example for clash default port
$Env:http_proxy="http://127.0.0.1:7890"
$Env:https_proxy="http://127.0.0.1:7890"
```

### path
