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



### [force `git pull`](https://stackoverflow.com/questions/1125968/git-how-do-i-force-git-pull-to-overwrite-local-files)

```bash
# reset
git reset --hard HEAD

# WARN: this delete all untracked files & dirs.
# git clean -f -d -n # dry-run
git clean -f -d

# pull
git pull
```



### remove a large file wrongly committed and left in git history

large file, even if removed in current branch, will be left in history and every clone suffers from it.

The best choice is never to upload any, but if you have committed, this will save you:

```bash
# FOLDERNAME is what you want to remove from ALL history
git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch FOLDERNAME" -- --all

# further clean
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now

# push to remote
git push --all --force
```



### embed mp4 in readme.md (github only)

You just edit the markdown file in github webpage, drag and drop your mp4 video to it, and it will work.

It only writes a URL into your markdown, but github will render it as a video:

```markdown
**A GUI for training/visualizing NeRF is also available!**

https://user-images.githubusercontent.com/25863658/155265815-c608254f-2f00-4664-a39d-e00eae51ca59.mp4
```



### reset

say you would like to reset to a previous commit.

```bash
# check log to get the commit reference
git log --oneline

# say you'll reset to xxxxx commit
# --hard will rewrite the file content
git reset --hard xxxxx

# after that, you want to return to your previous commit.
git reflog show

# you found your previous commit is called HEAD@{y}
git reset --hard HEAD@{y}

```

