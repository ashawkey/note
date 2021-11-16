# docker

### Install

Follow the official websites.



### Basic Operations

```bash
# image
docker image ls
docker images 
docker search <image>
docker pull <library/>hello-world

# run container
docker run hello-world
docker run ubuntu echo "Hello World"
docker run -it ubuntu bash # i=input,t=terminal, type exit or <C-d> to quit.
docker run -d ubuntu bash -c "yes" # d=detach
docker run -d --name <name> <image/cmd> # assign name (else generating random name)

# map data
# always map directories! single file mapping can be bug-prone.
docker run -v /host:/container -it ubuntu bash # /host is mapped to /container, all modifications in /container reflects to /host. 
docker run --rm -v "$PWD/":/var/www/html <image> # rm=remove container after finish running, volume=mapping directories.

docker kill <ID>
docker stop <ID>
docker start <ID>
docker top <ID>

# ps
docker ps # name, ID, port, ...
# log
docker logs <ID> # see stdout

# container
docker container ls # running
docker container ls --all # all 
docker container run ... # the same as docker run
docker rm <ID> # must stop first

# port mapping
docker run -P <image/cmd> # -P: inner port map to random high outer port.
docker run -p 5000:5000 <image/cmd> # -p: assign port mapping manually (out:in)
docker port <ID> <inner_port> # return outer_port
```



### Docker file

```dockerfile
FROM centos:6.7
MAINTAINER hawkey "hawkey@haw.com"

RUN /bin/echo "hello!"
```

```bash
# build
docker build -t <name:tag> . # . means the context! (dockerfile, and other needed files. Usually an empty folder.)
# all the files in . will be packed to docker container.

# build from a specific dockerfile
docker build -t <name:tag> -f dockerfile . # assign dockerfile

# add tag
docker tag <ID> <name:new_tag>

# run 
docker run <name:new_tag>
```

```dockerfile
FROM <base_image> # scratch = null image

RUN # create a layer! (at most 127 layers)
RUN <cmd> # shell mode
	RUN echo "Hello!"
RUN [exec params] # execute mode
	RUN apt update
	RUN mkdir test
RUN buildDeps='gcc wget make' # this is the right way to use RUN
	&& apt update \
	&& wget https://someurl/somefile.tar \
	&& tar -xvf somefile.tar \
	&& apt purge -y --auto-remove $buildDeps # clear layer dependency


```

```dockerfile
COPY <source_path> <image_path>
ADD <source_path or url> <image_path> # not recommended
CMD <cmd>
CMD [exec params]
# docker container has no daemon. all CMD/RUN is executed in the host machine!
CMD service nginx start # is totally wrong.
CMD <"nginx", "-g", "daemon off;">
ENTRYPOINT <cmd>
ENTRYPOINT [exec params]
# difference from CMD is: expose the command to 'docker run'
docker run mycmd -additional_flag # CMD:error, ENTRYPOINT:good
ENV <key> <value>
ENV <key1>=<val1> <key2>=<val2> ...
ARG <key>[=<val>] # same as ENV, but will be removed after built.
VOLUME <path> # mount data
EXPOSE <port> # only declare! use -p out:in to do mapping.
WORKDIR <path> # change working directory of building
RUN cd /workdir # totally wrong, no effect to later workdir.
USER <user> # change user of building
```



### publish image

```bash
# build the image locally.
docker built -t <name>:<tag> .

# login
docker login -u <user> 

# create a remote repository in dockerhub.
# e.g., the repo is <user>/<name>

# tag local image
docker tag <name>:<tag> <user>/<name>:<tag>

# push
docker push <user>/<name>:<tag>
```

