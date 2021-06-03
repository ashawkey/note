# docker for CUDA + OpenGL

### nvidia-container-runtime

It depends on the host's NVIDIA devices and **drivers**!!!

https://nvidia.github.io/nvidia-container-runtime/

```bash
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime

###
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

### 
sudo systemctl restart docker
```



### Dockerfile

```dockerfile
FROM ubuntu:18.04
# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

```

build it

```bash
sudo docker build -t glvnd-x:latest .
```



### run.sh

this provides double tunneling:

local --- ssh ---> host --- docker ---> container

```bash
# tunnel ssh X forwarding
DISPLAY_NUMBER=$(echo $DISPLAY | cut -d. -f1 | cut -d: -f2)
export DISPLAY=:${DISPLAY_NUMBER}

socat TCP4:localhost:60${DISPLAY_NUMBER} UNIX-LISTEN:/tmp/.X11-unix/X${DISPLAY_NUMBER} &

# --rm: Make the container ephemeral (delete on exit).
# -it: Interactive TTY.
# --gpus all: Expose all GPUs to the container.
# --net host: necessary... but why?
docker run \
  --rm \
  -it \
  --gpus all \
  --net host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v $HOME/.Xauthority:/root/.Xauthority \
  --hostname $(hostname) \
  -e QT_X11_NO_MITSHM=1 \
  glvnd-x \
  bash
```

