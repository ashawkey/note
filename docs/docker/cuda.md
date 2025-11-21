# Docker for CUDA

### [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Enable GPU in docker >= 19.03.

The **driver** version is dependent on the host !!!

But the CUDA version can change freely as long as the driver supports.

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Check that it worked!
docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi
#docker run --rm --gpus all nvidia/cuda:10.1-base nvidia-smi # you can change cuda version freely!
```


Note:

* Always monitor in host by `nvidia-smi`. In container it will not display correctly.

* Use specific GPU:

  ```bash
  docker run --gpus '"device=1,2"' 
  ```

  
