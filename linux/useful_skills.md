# Linux useful skills

### Copy with exclusion

```bash
# copy source to destination, excluding source/folder
rsync -av --progress source destination --exclude folder

```



### Replace string in all files under a folder

```bash
# replace 'oldtext' to 'newtext' in all files under dir
grep -rl oldtext dir | xargs sed -i 's/oldtext/newtext/g'

find dir -type f -exec sed -i 's/oldtext/newtext/g' {} +
```

