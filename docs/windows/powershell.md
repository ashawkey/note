## powershell



### profile (bashrc)

```powershell
# location
$profile
# C:\Users\haw\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1

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

