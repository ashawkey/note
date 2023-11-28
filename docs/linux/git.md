# Git tutorial

### Log in

```bash
git config --global user.name "ashawkey"
git config --global user.email "ashawkey1999@gmail.com"

# cache password
git config --global credential.helper cache
```



### Start Projects & Make Changes

##### file status lifespan

![Git 下文件生命周期图。](https://git-scm.com/book/en/v2/images/lifecycle.png)

* Newly created files are **Untracked files（未被追踪）**
* Modified files are inside **Changes not staged for commit （未暂存的修改）**
* `git add` does two things, both will make the file **staged: Changes to be committed（暂存区）**
  * add an untracked file 
  * stage a modified file

```bash
# init
git init [name]
git clone <url> [name]

# track/stage a file
git add [files]
git add *
git add --all

# ignore files
# use .gitignore

# delete file
rm [file] # remove from workspace (Changed not staged for commit)
git rm [file] # remove from stage & workspace
git rm --cached [file] # only remove from stage (in case you wrongly added some data file)

# rename file
git mv [old] [new]
	# this equals three cmds
	mv [old] [new]
	git rm [old]
	git add [new]

# show changes
git status
git status -s # compressed

# show detailed differences
git diff           # unstaged vs. workspace
git diff --cached  # or --staged, staged vs. workspace

# commit changes
git commit                 # open editor 
git config --global core.editor # set default editor
git commit -m "message" # in a line
git commit -a -m "message" # auto add all modified (tracked) files and commit (omit stage)
```



### History

```bash
git log # all commits in details
git log -p # all commits with diff
git log -2 # latest 2 commits

git log --stat # in short
git log --oneline # simple

git log --pretty=format:"%h %s" --graph # tree view
git log --since=2.weeks
git log --grep="keyword"
```



### Undo

```bash
# amend commit
git commit -m "init"
git add forgotten_file
git commit --amend # only one commit is recorded.

# unstage 
git reset HEAD -- [file] # (changes to be commited) -> (changes not staged for commit)

# undo modifications
git checkout -- [file] # dangerous operation. all changes to [file] will be lost.
					   # in fact this copies from the last commited state.

```



### Remote 

```bash
Bob   $ git clone /home/alice/project myrepo
Alice $ git pull /home/bob/myrepo master # merge 

Alice $ git fetch /home/bob/myrepo master # only fetch, don't merge
Alice $ git log -p HEAD..FETCH_HEAD # show diff, or use gitk

# two-dot: "show everything that is reachable from the FETCH_HEAD but exclude anything that is reachable from HEAD"
Alice $ gitk HEAD..FETCH_HEAD # show diff

# three-dot: "show everything that is reachable from either one, but exclude anything that is reachable from both of them".
Alice $ gitk HEAD...FETCH_HEAD

# define remote repo shorthand
Alice $ git remote add bob /home/bob/myrepo
Alice $ git fetch bob # bob/master
Alice $ gitk master..bob/master
Alice $ git merge bob/master

Bob   $ git pull 
Bob   $ git config --get remote.origin.url
Bob   $ git branch -r
```

```bash
# clone from remote
git clone <remote url> [name]
	# this equals: 
	cd [name] && git init
	git remote add <remote url>
	git pull origin
	
git remote # see which remote servers you have configured
		   #=> origin # default shortname given to the server you cloned from.
git remote -v # show urls of shortnames to fetch and push
		      # if you have multiple remotes, it will list all.
git remote show <shortname> # show details about a remote

# add new remote
git remote add <shortname> <remote url>
git fetch <shortname>

# rename remote
git remote rename <oldname> <newname>

# remove remote
git remote remove <shortname>

# fetch and pull
git fetch <shortname> # pull down data you don't have yet, but don't merge to locals
					  # default is to fetch from "origin"
git pull <shortname> # fetch and merge

# push
git push <shortname> <branch>
# eg. git push origin master

# merge unrelated histories (eg. different git projects)
git merge master --allow-unrelated-histories
```

A usual workflow:

```bash
# clone
git clone <repo>
# set upsteam
git remote add upstream <repo>
# update 
git fetch upstream
# merge 
git merge upstream/master
# commit & push
git commit -m "my-update"
git push origin master
# pull request

```



### Tag

```bash
git tag
```



### Alias

```bash
git config --global alias.ci commit # git ci == git commit
git config --global alias.st status # git st == git status
```



### Branching

**Branch is a pointer** pointing to a commit object. 

The default branch is called `master`. There is also a pointer `HEAD` pointing to your current branch.

When you commit, the current branch (and `HEAD`) will automatically point to the new commit.

`checkout` command will point `HEAD` to the specified branch.

```bash
### create and move
git branch dev # create a new branch "dev" at the current commit object.

git log --oneline --decorate # this shows commits with branch names
git branch # show branch information, and where you are.
git branch -v # also show last commit of each branch
git branch --merged
git branch --no-merged

git checkout dev # move to "dev"
git checkout -b dev # create and move to "dev", equals to `git branch dev && git checkout dev`

git switch dev # move to "dev"
git switch -c dev # create and move to "dev"

### merge
git merge dev # merge "dev" into "master" (when no conflict)
# if master and dev are on the same line, this is called a Fast-Forward merge, since there is no conflict to be solve, git only move master pointer to dev.
# if master and dev are on different forks of the repo, this is called a three-source merge. git will find the diverge point (latest common ancestor) and merge three commit objects together.

# when conflict
git status # Unmerged path, must be solved manually...
git mergetool

# delete branch
git branch -d dev # safely delete merged "dev", only merged branch can be deleted this way.
git branch -D dev # forcely delete unmerged "dev" 
```



### Remote Branching

```bash
### fetch from remote server "origin", this will mentain a remote branch called "origin/master", while you work at local branch "master".
git fetch [origin]
git branch -r # or --remote, show remote branches

# fetch and merge in one step
git pull [origin]

# push to remote 
git push origin master

### sometimes the remote origin got a new branch "exp", and you want to move to that branch:
git fetch
git checkout origin/exp
# at this time, we entered a "detached HEAD" mode, which means there is no branch pointing to this commit at the local machine (origin/exp is not a local branch), and only a HEAD pointer pointing here. Usually then we want to make a local branch here too.
git checkout -b exp # local exp branch.
git log --graph # finished.

# one line version
git checkout -t origin/exp # track, equals above 2 cmds
git checkout exp # even simpler, equals above cmd

# push to origin/exp
git push origin/exp exp
```



### Rebase Branch

Another method to merge branches.

![分叉的提交历史。](https://git-scm.com/book/en/v2/images/basic-rebase-1.png)

```
git checkout experiment
git rebase master
```

![将 `C4` 中的修改变基到 `C3` 上。](https://git-scm.com/book/en/v2/images/basic-rebase-3.png)

```bash
git checkout master
git merge experiment
```

![master 分支的快进合并。](https://git-scm.com/book/en/v2/images/basic-rebase-4.png)



While normal merge will generate:

![通过合并操作来整合分叉了的历史。](https://git-scm.com/book/en/v2/images/basic-rebase-2.png)



### Ignore

`.gitignore`

```
workspace
*.log
!important.log
```

```bash
# ignore tracked files
git rm --cached <file>
git rm -r --cached <folder>
```



### Submodule

Add a repo (at `url`) to the current repo's `dest_path` as a submodule:

```bash
git submodule add <submodule_url> <dest_path>
```

This will automatically track the submodule's version by creating a `.gitmodules` file.



### Stash

Use situation: You want to switch to another branch, but there are uncommitted modifications in the current branch.

If you checkout now, you'll get errors:

```bash
error: Your local changes to the following files would be overwritten by checkout:
        .gitignore
Please, commit your changes or stash them before you can switch branches.
```

And if you don't want to commit, you'll need stash:

```bash
git stash
# Saved working directory and index state WIP on main: f559623 xxx
# HEAD is now at f559623 xxx

git status
# now the uncommited modificaitons are reverted.

# do your works now...
# e.g., git checkout dev

git stash list
# stash@{0}: WIP on main: f559623 xxx

git stash apply # == git stash apply stash@{0}
# everything is back again!

git stash drop stash@{0}
# delete the stash since it is useless now.
```

