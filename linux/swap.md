# swap

### check swap

use `htop`.



### create swap

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```



### delete swap

```bash
sudo swapoff /swapfile
sudo rm  /swapfile
```



### `kswapd0` high CPU usage.

This is usually caused by unmounted Swap if there is not enough memory.

See the second answer [here](https://askubuntu.com/questions/259739/kswapd0-is-taking-a-lot-of-cpu).