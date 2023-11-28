## Tyro

Although `omegaconf`+`argparse` is great for a research project, it's not suitable for a python package CLI (we have no place to put all the config files).

With `tyro`, we can write a package CLI elegantly!



### Function

The simplest use case, turn a function into a CLI.

```python
import tyro

def main(
    x: int, # required
    path: str = 'logs', # with a default value
):
    print(x, path)

if __name__ == '__main__':
    tyro.cli(main)
```

```bash
python main.py --help
python main.py --x 1
python main.py --x 1 --path wow
```



### Dataclass

For more complex configs.

A `dataclass` is a specialized object for holding static data:

```python
from dataclasses import dataclass

@dataclass
class A:
    #x # without type specification it will throw error!
    x: str # ok
    y: int = 1 # with default value
    
#print(A.x) # error
print(A.y) # 1

a = A(x='x')
print(a.x, a.y) # 'x', 1
```

We can use `tyro` to parse a `dataclass`:

```python
import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Union
import enum

class Color(enum.Enum):
    red = enum.auto()
    green = enum.auto()
    blue = enum.auto()

@dataclass
class Options:
    # NOTE: must place positional values > required vales > default values
    # NOTE: tyro will detect comments (after > above) as the help string.
    
    # positional value
    x: tyro.conf.Positional[int]

    # required value
    y: int 

    # required bool, "--flag1 True/False"
    flag1: bool 

    # default values
    path: str = 'logs' 

    # variable-length
    shape: Tuple[int, ...] = (64,)

    # multi-value
    size: Tuple[int, int] = (64, 64)

    # choice by enum
    color: Color = Color.red

    # choice by Literal
    color2: Literal['red', 'green', 'blue'] = 'red'

    # bool default to False, use "--flag2" to set True
    flag2: bool = False 

    # bool default to True, use "--no-flag3" to set False
    flag3: bool = True 

    # mixed type by union
    int_or_str: Union[int, str] = 0


if __name__ == '__main__'    :
    opt = tyro.cli(Options)
    print(opt)
```

```bash
python main.py --help

# output
usage: main.py [-h] --x INT --flag1 {True,False} [--path STR] [--shape INT [INT ...]] [--size INT INT]
               [--color {red,green,blue}] [--color2 {red,green,blue}] [--flag2] [--no-flag3]
               [--int-or-str INT|STR]

╭─ positional arguments ───────────────────────────────────────────────────────────────────────╮
│ INT                     positional parameters (required)                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ arguments ──────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                      │
│ --x INT                 required value (required)                                            │
│ --flag1 {True,False}    required bool, "--flag1 True/False" (required)                       │
│ --path STR              default values (default: logs)                                       │
│ --shape INT [INT ...]   variable-length (default: 64)                                        │
│ --size INT INT          multi-value (default: 64 64)                                         │
│ --color {red,green,blue}                                                                     │
│                         choice by enum (default: red)                                        │
│ --color2 {red,green,blue}                                                                    │
│                         choice by Literal (default: red)                                     │
│ --flag2                 bool default False, use "--flag2" to set True (sets: flag2=True)     │
│ --no-flag3              bool default True, use "--no-flag3" to set False (sets: flag3=False) │
│ --int-or-str INT|STR    mixed type by union (default: 0)                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```



### Hierarchical Configs

For nesting configs.

```python
import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Union

@dataclass
class OptimizerOption:
    type: Literal['adam', 'sgd'] = 'adam'
    lr: float = 1e-3
    
@dataclass
class Options:
    # nested option: just declare the type
    optimizer: OptimizerOption
    # other options
    seed: int = 0
    iterations: int = 3000
    

if __name__ == '__main__'    :
    opt = tyro.cli(Options)
    print(opt)
```

```bash
python main.py --help

# output
usage: main.py [-h] [--optimizer.type {adam,sgd}] [--optimizer.lr FLOAT] [--seed INT] [--iterations INT]

╭─ arguments ─────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --seed INT              other options (default: 0)      │
│ --iterations INT        other options (default: 3000)   │
╰─────────────────────────────────────────────────────────╯
╭─ optimizer arguments ───────────────────────────────────╮
│ nested option: just declare the type                    │
│ ────────────────────────────────────────                │
│ --optimizer.type {adam,sgd}                             │
│                         (default: adam)                 │
│ --optimizer.lr FLOAT    (default: 0.001)                │
╰─────────────────────────────────────────────────────────╯
```



### Subcommands

```python
import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Union

@dataclass
class Train:
    type: Literal['adam', 'sgd'] = 'adam'
    lr: float = 1e-3
    seed: int = 0
    iterations: int = 3000
    
@dataclass
class Test:
    # other options
    resolution: int = 1024
    

if __name__ == '__main__':
    # pass in a Union of all configs of subcommands
    opt = tyro.cli(Union[Train, Test])
    print(opt) # you'll have to decide whether opt is Train or Test by isinstance...
```

```bash
python main.py --help

# output
usage: main.py [-h] {train,test}

╭─ arguments ─────────────────────────────────────────╮
│ -h, --help          show this help message and exit │
╰─────────────────────────────────────────────────────╯
╭─ subcommands ───────────────────────────────────────╮
│ {train,test}                                        │
│     train                                           │
│     test                                            │
╰─────────────────────────────────────────────────────╯

python main.py train --help

# output
usage: main.py train [-h] [--type {adam,sgd}] [--lr FLOAT] [--seed INT] [--iterations INT]

╭─ arguments ─────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --type {adam,sgd}       (default: adam)                 │
│ --lr FLOAT              (default: 0.001)                │
│ --seed INT              (default: 0)                    │
│ --iterations INT        (default: 3000)                 │
╰─────────────────────────────────────────────────────────╯
```



### Subcommands for overriding default configs

This can reach the same effect as using multiple config files with `omegaconf`, but it's a little tricky...

```python
import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Union, Dict

@dataclass
class Options:
    name: str
    type: Literal['adam', 'sgd'] = 'adam'
    lr: float = 1e-3
    seed: int = 0
    iterations: int = 3000

# different default configs
config_defaults: Dict[str, Options] = {}
config_descriptions: Dict[str, str] = {}

config_defaults['A'] = Options(name='A')
config_descriptions['A'] = 'this is setting A'

config_defaults['B'] = Options(name='B', lr=1e-4) # different default lr
config_descriptions['B'] = 'this is setting B'

if __name__ == '__main__':
    opt = tyro.cli(tyro.extras.subcommand_type_from_defaults(config_defaults, config_descriptions))
    print(opt)
```

```bash
python main.py --help

# output
usage: main.py [-h] {A,B}

╭─ arguments ───────────────────────────────────────╮
│ -h, --help        show this help message and exit │
╰───────────────────────────────────────────────────╯
╭─ subcommands ─────────────────────────────────────╮
│ {A,B}                                             │
│     A                 this is setting A           │
│     B                 this is setting B           │
╰───────────────────────────────────────────────────╯

python main.py A --help

# output
usage: tmp_tyro.py A [-h] [--name STR] [--type {adam,sgd}] [--lr FLOAT] [--seed INT] [--iterations INT]

this is setting A

╭─ arguments ─────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --name STR              (default: A)                    │
│ --type {adam,sgd}       (default: adam)                 │
│ --lr FLOAT              (default: 0.001)                │
│ --seed INT              (default: 0)                    │
│ --iterations INT        (default: 3000)                 │
╰─────────────────────────────────────────────────────────╯
```

