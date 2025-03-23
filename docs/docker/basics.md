# docker basics

### install

Follow https://docs.docker.com/engine/install/ubuntu/

```bash
### ubuntu
# remove old
sudo apt-get remove docker docker-engine docker.io containerd runc

# install from repo
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io  

# verify
sudo docker run --rm hello-world
```

Or install from [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/):

```bash
export DOWNLOAD_URL="https://mirrors.tuna.tsinghua.edu.cn/docker-ce"

curl -fsSL https://raw.githubusercontent.com/docker/docker-install/master/install.sh | sh
```

You can control docker daemon by:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

By default, only root user (`sudo`) can use docker. A docker group can be created to avoid this, but anyone in the docker group equals a sudoer.

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```


### docker hub mirrors

Add in `/etc/docker/daemon.json`:

```json
{
  "registry-mirrors": [
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```


### command line

* get image from docker hub:

  Image is a pre-built system to run in docker.

  ```bash
  docker pull user/repo[:tag]
  ```

* list local images:

  ```bash
  # ls local images (top layer images)
  docker image ls
  
  # example otuput:
  # REPOSITORY     TAG          IMAGE ID       CREATED        SIZE
  # hello-world    latest       301bfc30ea2b   43 hours ago   13.3kB
  
  
  # ls dangling images (deprecated images, displayed as <none>:<none>)
  docker image ls -f dangling=true
  # auto clean dangling images
  docker image prune
  
  # ls all local images (including middle layer images/dependencies, do not delete them!)
  docker image ls -a
  
  # ls matched repos
  docker image ls ubuntu
  
  # only ls IMAGE ID (can be used with rm)
  docker image ls -q
  ```

* remove local images

  ```bash
  # rm a local image
  docker image rm <name[:tag]/image_ID>
  
  # rm all images named ubuntu*
  docker image rm -f $(docker image ls -q ubuntu)
  
  # prune to free storage
  docker system prune -f
  ```

* run & manage container

  Container is the instantiation of an image.

  ```bash
  ### run a command and exit.
  docker run ubuntu:18.04 /bin/echo "Hello!"
  
  ### run interactive container
  # --interactive, --tty. should follow nothing, or a specific shell
  docker run -it ubuntu:18.04 [/bin/bash]
  # enter shell
  echo "Hello!"
  exit
  # exit shell
  
  # auto remove the container when exit
  docker run --rm -it ubuntu:18.04
  
  ### run detached container 
  
  # the command itself must be non-terminating!
  docker run -d [--name test] ubuntu:18.04 /bin/bash -c "while true; do echo hello world; sleep 1; done"
  # will print the <ID> of this container
  
  # this will still exit after finishing the command
  docker run -d ubuntu:18.04 /bin/bash -c "echo hello world"
  
  # detached interactive:
  docker run -dit ubuntu:18.04 
  
  ### port mapping (--publish)
  # single port mapping
  docker run -d -p <host>:<container> nginx
  # random port mapping
  docker run -d -P nginx
  # use host network (all ports are mapped identitcally)
  docker run --network host nginx
  
  ### volume mapping (--volume)
  docker run -v /host:/container -it ubuntu
  
  ### docker container
  
  # only ls running containers
  docker container ls
  # example output:
  # CONTAINER ID   IMAGE    COMMAND    CREATED     STATUS          PORTS     NAMES
  # if name not assigned, it will be randomly generated like `elastic_knuth`
  
  # ls all containers (recently stopped)
  docker container ls -a
  
  # print logs from detached container
  docker container logs <ID/name>
  
  # terminate a running container
  docker container stop <ID/name>
  
  # start a terminated container
  docker container start <ID/name>
  
  # restart a running container
  docker container restart <ID/name>
  
  ### attach to the current shell of a detached container
  docker attach <ID/name>
  # enter shell
  # if exit from this shell, the container will stop too.
  
  ### attach by entering a new shell.
  docker exec -it <ID/name> bash
  # enter shell
  # if exit from this shell, the container will not stop!
  
  ### remove terminated container
  docker container rm <ID/name>
  
  # auto remove all termintaed containers
  docker container prune
  ```

* other utils

  ```bash
  # ps of all running containers
  docker ps 
  
  # ps of all containers
  docker ps -a
  
  # filtered ps
  docker ps --filter "name=nostalgic"
  
  # top 
  docker top <ID/name>
  ```
  
* commit to image

  ```bash
  # run a detached nginx container
  docker run --name webserver -d -p 80:80 nginx
  
  # enter the shell of the container
  docker exec -it webserver bash
  # begin shell: modify the index file
  echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
  exit
  # end shell
  
  # check changes by diff 
  docker diff webserver
  
  # commit changes
  docker commit webserver nginx:v2 [--message "change index"]
  
  # check the new image
  docker image ls nginx
  
  # check history
  docker history nginx:v2
  
  # now you can run this new image
  docker run --name webserver2 -d -p 81:81 nginx:v2
  ```

  Never use commit to build a image! The image built from commit is a black box, and may contain many redundant layers.

  Always use `Dockerfile` to build a image!

  
### Dockerfile

To build an image, we should first create an empty folder and create a `Dockerfile`:

```bash
mkdir repo
cd repo
vim Dockerfile
```

edit it:

```dockerfile
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

Then build it:

```bash
# image name is nginx, tag is v3, context is pwd
docker build -t nginx:v3 .
```

You can even build from github repository:

```bash
# the repo should contain a Dockerfile
docker build -t hello-world https://github.com/docker-library/hello-world.git#master:amd64/hello-world
```

Push to registry:
```bash
# login to your registry
docker login registry

# tag your image with the name in remote registry
docker image tag myimage:version registry/myname/myimage:version

# you should see both images (share the same ID)
docker image ls

# then push it
docker image push registry/myname/myimage:version
```

Detailed command:

* from base image

  ```dockerfile
  FROM ubuntu
  FROM scratch
  ```

* copy from the **context folder** to container

  ```dockerfile
  COPY [--chown=<user>:<group>] <srcfiles> <dstdir>
  
  # copy files
  # if dstdir (/myfiles/) do not exit, it will be created automatically.
  # the last / is necessary! 
  COPY index.html /myfiles/
  
  # copy folder is however tricky: by default it copy the content, not the folder itself
  COPY folder /myfiles/ # cp folder/* /myfiles/
  COPY folder /myfiles/folder # cp folder /myfiles/
  ```

* add: advanced copy, but not recommended.

  it can automatically decompress files, download from URL, ...

  ```dockerfile
  # copy then `tar -zxvf` 
  ADD ubuntu-xenial-core-cloudimg-amd64-root.tar.gz /
  
  # it can replace all usage of COPY
  COPY [--chown=<user>:<group>] <srcfiles> <dstdir>
  ```

* run command when building the image

  ```bash
  # run command
  RUN apt update \
      && apt upgrade \
      && apt install -y nginx
  
  # always clear the workspace!
  RUN set -x; buildDeps='gcc libc6-dev make wget' \
      && apt-get update \
      && apt-get install -y $buildDeps \
      && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
      && mkdir -p /usr/src/redis \
      && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
      && make -C /usr/src/redis \
      && make -C /usr/src/redis install \
      && rm -rf /var/lib/apt/lists/* \
      && rm redis.tar.gz \
      && rm -r /usr/src/redis \
      && apt-get purge -y --auto-remove $buildDeps
  ```

  Use RUN sparingly! Each RUN will create a layer, and we should use as less layer as possible.
  
* run command when container start

  ```bash
  # shell mode
  CMD <shell command>
  CMD echo hello
  
  # exec mode
  CMD ["file", "args", ...] # must use "", not '', since it will be converted to json.
  CMD ["bash", "-c", "echo hello"]
  
  # entrypoint mode
  # define the base command, all later args will be appended to it.
  # example:
  CMD [ "curl", "-s", "http://myip.ipip.net" ]
  # docker run myip [OK]
  # docker run myip -i [ERROR]
  # instead, if we use
  ENTRYPOINT [ "curl", "-s", "http://myip.ipip.net" ]
  # docker run myip [OK]
  # docker run myip -i [OK]
  ```


* environment variables

  ```dockerfile
  # env: persistent, can still be used in running containers.
  ENV <key> <val>
  ENV <k1>=<v1> <k2>=<v2> ...
  
  ENV DEBUG=on
  RUN echo $DEBUG
  
  # arg: non-persistent, only available in dockerfile
  ARG <key>=<val>
  
  # only available in FROM
  ARG DOCKER_USERNAME=library
  FROM ${DOCKER_USERNAME}/alpine
  # should reassign
  ARG DOCKER_USERNAME=library
  RUN set -x ; echo ${DOCKER_USERNAME}
  ```

* create a anonymous volume.

  It is just a declaration to create the folder.

  ```dockerfile
  VOLUME <path>
  
  VOLUME /data
  ```

  In practice, we need to use `-v` to replace these anonymous volumes to make data persistent.

  ```bash
  # this will use /host to replace to anonymous volume.
  docker run -it -v /host:/data ubuntu
  ```

* expose port

  It is just a declaration that the port can be opened.

  ```dockerfile
  EXPOSE <port> [<port2> ...]
  ```

  In practice, we need to use `-p` to map the port.

  ```bash
  docker run -it -p 80:80 nginx
  ```

* change work directory

  ```dockerfile
  # create the folder, and later commands will be executed here.
  WORKDIR <path>
  
  # default workdir is root dir
  WORKDIR /
  
  # example
  WORKDIR /a # current workdir is /a/
  WORKDIR b # current workdir is /a/b/
  RUN touch c # create file at /a/b/c
  ```

* change user

  ```bash
  # must create the user first
  RUN groupadd -r redis && useradd -r -g redis redis
  
  # change to user redis for later commands.
  USER redis
  RUN [ "redis-server" ]
  ```

* add metadata

  ```dockerfile
  LABEL <key>=<val> ...
  ```

