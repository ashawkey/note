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
    # NOTE: all required values should be put ahead of default values!
    # NOTE: tyro will detect comments (after > above) as the help string.

    # required value
    x: int 

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



