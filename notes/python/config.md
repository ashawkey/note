### argparse

when you just have a simple project, and don't want to write `yaml` config files.

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=None, type=str)
opt = parser.parse_args()
```


### omega conf

for large projects, when you need different configs (`yaml`) for different running.

```bash
pip install omegaconf
```

Omega conf basics:

```python
from omegaconf import OmegaConf

# create
opt = OmegaConf.create() # empty DictConfig
opt = OmegaConf.create({'x': 1}) # from dict
opt = OmegaConf.load('config.yaml') # from yaml

# access
opt.x
opt['x']
opt.get('y', 10) # default value if not exists

# mandatory values
opt = OmegaConf.create({'x': '???'}) # ??? means this value must be provided before access
opt.x # Traceback...

# use environment variables
opt = OmegaConf.create({'x': '{oc.env.USER}'}) # {oc.env.XXX} is a built-in resolver
```

Use it in practice (allow loading basic `yaml` config and override from command line)

```bash
./configs
  - base.yaml # x: 1
  - dev.yaml
main.py  
```

the `main.py` script:

```python
import argparse
from omegaconf import OmegaConf

def train(cfg): pass
def test(cfg): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--gpu", default="0", help="GPU ID")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    args, extras = parser.parse_known_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # override default config from cli
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras)) # 
    
    if args.train:
        train(cfg)
    elif args.test:
        test(cfg)
    
if __name__ == '__main__':
    main()
```

use it in command line:

```bash
# base config
python main.py --config configs/base.yaml --train --gpu 0
# override config
python main.py --config configs/base.yaml --train --gpu 0 x=2 # now x is 2
```


### tyro

An even better library, satisfying all the need, check the separate note [here](./tyro.md).