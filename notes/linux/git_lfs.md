# git-lfs

Ref: [tutorial](https://zzz.buzz/zh/2016/04/19/the-guide-to-git-lfs/#%E5%B8%B8%E7%94%A8-git-lfs-%E5%91%BD%E4%BB%A4)


### install

Download the [releases](https://github.com/git-lfs/git-lfs/releases) and follow platform specific instructions.


### usage

```bash
# download large files under a repo using lfs.
git lfs pull

# initalize lfs in a git repo
git lfs install

# track certain files (png for example)
git lfs track *.png

# then it's safe to push, it will automatically handle all pngs as lfs
git add *
git commit -m 'lfs'
git push
```

