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

you can also fully revert the commit and commit again

```bash
git reset --hard HEAD~1
```





### undo/reset anything using reflog

`git reflog` keeps track of all your git commands, which is very useful to undo/redo.

```bash
git reflog
# output is like:
# c4ddb13 (HEAD -> master, origin/master, origin/HEAD) HEAD@{0}: pull: Fast-forward
# b9a8d4f HEAD@{1}: pull: Fast-forward
# 040fa10 HEAD@{2}: pull: Fast-forward
```

Notation:

* `HEAD` is the current state.
* `HEAD~<n>` means the n-th ancestor of HEAD (back trace through commit history/graph)
* `HEAD@{n}` means the n-th history command through `reflog` (it's different from ancestor!)

After finding the aimed commit, we can reset by:

```bash
# soft: revert commits (but changes are still staged, i.e., after `git add`)
git reset --soft <commit>

# mixed: revert commits and unstage changes.
git reset --mixed <commit>

# hard: revert commits, undo changes. Your branch will be the same as <commit> again.
git reset --hard <commit>
```



### command line diff

If you cannot use VS Code or Github.

```bash
# see changes (before git add)
git diff 
# after git add
git diff --staged
```






### checkout a history commit, and also update all submodules to the corresponding commits

```bash
git log # find your commit
git log --reverse # old-to-new history

git checkout <hash> # only checkout main repo
git submodule update --recursive # checkout all submodules too.
```




### [force `git pull`](https://stackoverflow.com/questions/1125968/git-how-do-i-force-git-pull-to-overwrite-local-files)

```bash
# reset all changes
git reset --hard HEAD

# WARN: this delete all untracked files & dirs.
# git clean -f -d -n # dry-run
git clean -f -d

# pull
git pull
```



### remove a large file wrongly committed and left in git history

large file, even if removed in current branch, will be left in history and every clone suffers from it.

The best choice is never to upload any, but if you have committed, the best way is to soft reset these commits and re-commit again, although this will squash all the later commits:

```bash
# find the last commit before adding the large files
git reflog

# soft reset
git reset --soft <commit id>

# delete the large files, add all other files and commit again.
rm <big_file>
git add *
git commit -m 'squash commits'
```



### embed mp4 in readme.md (github only)

You just edit the markdown file in github webpage, drag and drop your mp4 video to it, and it will work.

It only writes a URL into your markdown, but github will render it as a video:

```markdown
**A GUI for training/visualizing NeRF is also available!**

https://user-images.githubusercontent.com/25863658/155265815-c608254f-2f00-4664-a39d-e00eae51ca59.mp4
```




### manage remote repo url

```bash
# change remote origin
git remote set-url origin [new_repo_url]

# or you can add it as another remote
git remote set-url origin2 [new_repo_url]
git push -u origin2 main
```



### un-ignore specific files

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



### multiple accounts in the same shell

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



### art of merge request

In a collaborated project where you need to work on your branch, but keep updated with the main branch:

```bash
# create your branch from main
git checkout -b mybranch
# work on your branch and commits... maybe also push to your remote branch.

# now you want to merge to main, but other people also merged many things already. 
# fetch remote changes
git fetch origin
# rebase (replay) your changes to origin/main (ours are incoming)
git rebase origin/main
# or origin/main's changes into your code (they are incoming), sometimes this is easier
git merge origin/main

# now your commits and other people's commits are both applied, but this has diverged from your remote branch since rebase will add other people's commits (if you have pushed before), so you need to force update your remote branch
git push origin --force
```

You may want to squash your commits before rebasing:

```bash
# reset to that earliest+1 commit you want to squash
git reset --soft HEAD~3 # go back 3 commit from head (will squash HEAD, HEAD~, HEAD~2)
git reset --soft <commit id> # will squash commits AFTER this commit

# now simply commit all changes
git commit -m 'squash!'
```

Another case is that you are working on a branch and submitted MR to origin/main, which is squash-merged later, but you are still working on the same branch with some new commits. (The best practice is not to work on the same branch until its MR is merged, but sometimes this does happen...)

Now the origin/main's commit history has diverged from your branch (as it contains only some of your commits but squashed). To fix this:

```bash
# sync local main first
git checkout main
git fetch origin
git pull

# create a new branch for MR
git checkout -b mr_branch

# merge your old branch to it
git merge old_branch

# now mr_branch contains exactly one commit from origin/main with your new commits squashed. we can submit MR with this branch now!
```



### pushed a branch with wrong name

We need to create a new branch with correct name, delete the old branch and push new branch again.

```bash
# rename (move) local branch
git branch -m <correct_name>
# push it to remote
git push origin -u <correct_name>

# delete a remote branch.
git push origin --delete <wrong_name> 
# delete a local branch.
git branch -d <wrong_name>
```



### merge part of your changes to main

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

