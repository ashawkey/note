### I forget `--recursive` when cloning

```bash
git submodule update --init --recursive
```



### undo `git pull`

**git pull** performs two operations:

```bash
git fetch
git merge
```

To undo it:

```bash
# find the commit id
$ git reflog
c4ddb13 (HEAD -> master, origin/master, origin/HEAD) HEAD@{0}: pull: Fast-forward
b9a8d4f HEAD@{1}: pull: Fast-forward
040fa10 HEAD@{2}: pull: Fast-forward
6e544a4 HEAD@{3}: pull: Fast-forward
56bab4d HEAD@{4}: pull: Fast-forward
7e1f9de HEAD@{5}: pull: Fast-forward
092a51f HEAD@{6}: pull: Fast-forward
2cb2adc HEAD@{7}: pull: Fast-forward
eb28445 HEAD@{8}: pull: Fast-forward
fd23ebc HEAD@{9}: pull: Fast-forward
5d31b72 HEAD@{10}: pull: Fast-forward
c860334 HEAD@{11}: clone: from https://github.com/ashawkey/ashawkey.github.io.git

# reset (e.g., to last commit, i.e., HEAD@{1})
git reset --hard b9a8d4f 

# also, you can use time:
git reset --hard master@{5.days.ago}
# like `10.minutes.ago`, `1.hours.ago`, `1.days.ago`
```

Improvement: **use `git pull --rebase` instead of `git pull`!**

[It also syncs server change](https://gitolite.com/git-pull--rebase).

