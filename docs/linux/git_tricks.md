### I forget adding recursive when cloning

```bash
git submodule update --init --recursive
```


### add more things to the last commit (amend)

```bash
# add the modifications
git add *
# amend
git commit --amend --no-edit

# this will also allow you to change the commit message
git commit --amend
```


### command line diff

If you cannot use VS Code or Github.

```bash
# see changes (before git add)
git diff 
# after git add
git diff --staged
```


### modify the last commit

```bash
# reset to the status before last commit
git reset HEAD~1
# now correct your mistakes...
...
# redo the commit
git add *
git commit -c ORIG_HEAD
```


### revert a commit

```bash
# find the commit hash
git log
# revert it
git revert <hash>
```


### checkout a history commit, and also update all submodules to the corresponding commits

```bash
git log # find your commit
git log --reverse # old-to-new history

git checkout <hash> # only checkout main repo
git submodule update --recursive # checkout all submodules too.
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


### reset to a history commit

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


### change remote repo url

```bash
# change remote
git remote set-url origin [new_repo_url]

# then you can normally push!
```


### push a new branch to remote

```bash
# to local branch
git checkout -b <branch>

# push branch to remote
git push -u origin <branch>
#To https://github.com/ashawkey/svd_nerf.git
# * [new branch]      arithm -> arithm
#Branch arithm set up to track remote branch arithm from origin.

```


### gitignore un-ignore specific files

say you want to exclude everything in `datasets/` except `datasets/splits/*`.`

```bash
#datasets # this will not work!
datasets/* # the /* matters!

!datasets/splits/
```


### use ssh for git command (avoid permission denied error)

You need to generate a SSH key for your machine and add it to github.

```bash
cd  ~/.ssh
ssh-keygen -t ecdsa -b 521 -C "your_email@example.com"

cat id_ecdsa.pub
```

Then copy the public key and add it in https://github.com/settings/keys

Now you should be able to clone through ssh!


### fetch branch from a forked repo without clone

```bash
git remote add theirusername https://github.com/theirusername/reponame
git fetch theirusername
git checkout -b mynamefortheirbranch theirusername/theirbranch
```

### Multiple accounts in the same shell

This happens when you want to switch to another github/gitlab account for a specific repo:

* Set local user name and email:

```bash
git config --local user.email "subaccount@gmail.com"
git config --local user.name subaccount
```

* Normally init repo and commit.
* Set remote URL in this format (**add `USERNAME@` before github.com!**)

```bash
git remote set-url origin https://USERNAME@github.com/USERNAME/PROJECTNAME.git
```

* Create a PAT through web client. (Settings > developer settings > personal access tokens)
* Push, enter your PAT through the GUI.



### Rebase remote commits from others

In a collaborated project where you need to work on your branch, but keep updated with the main branch:

```bash
# create your branch from main
git checkout -b mybranch
# work on your branch and commits... maybe also push to your remote branch.

# now you want to merge to main, but other people also merged many things already. 
# fetch remote changes
git fetch origin
# rebase remote changes on main to your local branch
git rebase origin/main
# or if you prefer merge, which might be easier
git merge origin/main
# now your commits and other people's commits are both applied, but this has diverged from your remote branch since rebase will add other people's commits (if you have pushed before), so you need to force update your remote branch
git push origin --force
```



### Pushed a branch with wrong name

We need to create a new branch with correct name, delete the old branch and push new branch again.

```bash
# rename (move) local branch
git branch -m <correct_name>
# push it
git push origin -u <correct_name>
# delete wrongly named and pushed remote branch.
git push origin --delete <wrong_name> 

# delete a local branch.
git branch -d <wrong_name>
```





### Merge part of your changes to main

This is not easy, especially if these changes are in many different commits, and between them you have commits that you do not want to merge to main.

We need to wipe the commit history and re-commit those needed changes:

```bash
# create new branch from current branch
git checkout -b mybranch_to_merge

# make sure you have synced all changes from origin/main !!!
# if this is impossible, at least fetch them

# mixed reset to origin/main
git reset --mixed origin/main
# now you see all the changes you have done, all unstaged.
# add back those changes you want to merge, and commit
# push to remote and create MR

# if you have unmerged changes from origin/main, you may want to undo those changes
git checkout <file/folder>
```

>  About `git reset`:
>
> ```bash
> # soft: revert commits (but changes are still staged, i.e., after `git add`)
> git reset --soft origin/main
> 
> # mixed: revert commits and unstage changes.
> git reset --mixed origin/main
> 
> # hard: revert commits, undo changes. Your branch will be the same as origin/main again.
> git reset --hard origin/main
> ```
