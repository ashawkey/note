# PyPi publish



### prepare `setup.py`

A simple example:

```python
from setuptools import setup

if __name__ == '__main__':
    setup(
        name="numpytorch",
        version='0.1.0',
        description="Monkey-patched numpy with pytorch syntax",
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ashawkey/numpytorch',
        author='kiui haw',
        packages=['numpytorch',],
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3 ',
            'License :: OSI Approved :: MIT License',
        ],
        keywords='tensor manipulation, machine learning, deep learning',
        install_requires=[
            'numpy>=1.20',
            'forbiddenfruit',
        ],
    )
```

`setup.cfg` is a static alternative of `setup.py`. Usually we don't need it. (although `setuptools` aims to transfer to the static one.)





### `pytest` for unit test

```python
pip install -U pytest
```

Prepare the folders as:

```bash
- pkg
	- __init__.py
	- ...
- tests
	- test_1.py
	- test_2.py
	- ...
setup.py
```

Install the package as editable (just link the package to the original location, basically meaning any changes to the original package would reflect directly in your environment.):

```bash
# -e, --editable
pip install -e .
```

Write the `test.py` as:

```python
import pkg

def test_func1():
    assert (...)

def test_func2():
    assert (...)
    
def TestClass:
    def test_method1():
        assert (...)
```

Call `pytest` at the current directory, it will automatically locate `tests/test_*.py`:

```bash
# run all tests
pytest
# quite run
pytest -q
# stop at first error
pytest -x

# test matching keyword
pytest -k func
pytest -k Class.method
```



### Publishing

