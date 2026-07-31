# UV

## Why UV? Why not Conda?

* Modern design, standards-first, especially for pure-python lightweight project.
* UV is efficient in downloading packages, but NOT efficient in disk storage. It creates `.venv` per project and duplicate wheels, so big wheels like `pytorch` is not recommended.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# or use pip
pip install uv

# enable shell completion (assuming bash)
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc
```

## Usage

Instead of directly call `python`, use `uv run` to activate the uv environment.

```bash
# run example.py in the current directory
# it will detect the pyproject.toml and install dependencies automatically.
uv run example.py

# run but not detect or install dependencies
uv run --no-project example.py

# run with explicit dependencies (will be installed to a temporary virtual environment in `UV_CACHE_DIR`)
uv run --with rich example.py
```

When working in a **project** directory (`pyproject.toml`), uv will create a `.venv` to install the dependencies.
Otherwise, it uses `UV_CACHE_DIR=~/.cache/uv` for a temporary virtual environment.

```bash
# show the cache directory
uv cache dir
# /home/kiui/.cache/uv

# clear the cache (delete all downloaded packages, and temporary virtual environments)
uv cache clean
# Clearing cache at: .cache/uv
# Removed 99899 files (17.3GiB)
```

Detect pythons:
```bash
uv python list
# cpython-3.11.14-linux-x86_64-gnu                <download available>
# cpython-3.11.7-linux-x86_64-gnu                 anaconda3/bin/python3.11
# cpython-3.11.7-linux-x86_64-gnu                 anaconda3/bin/python3 -> python3.11
# cpython-3.11.7-linux-x86_64-gnu                 anaconda3/bin/python -> python3.11
# cpython-3.10.15-linux-x86_64-gnu                .local/share/uv/python/cpython-3.10.15-linux-x86_64-gnu/bin/python3.10
# cpython-3.10.12-linux-x86_64-gnu                /usr/bin/python3.10
# cpython-3.10.12-linux-x86_64-gnu                /usr/bin/python3 -> python3.10
# ...
```

Working by project:
```bash
# create a template project (with pyproject.toml, .gitignore, etc.)
uv init my_project
cd my_project

# add dependency (add to pyproject.toml)
# Avoid `requirements.txt` and `uv pip`, use `pyproject.toml` and `uv add`.
uv add rich

# remove dependency
uv remove rich

# run the project (will install dependencies from pyproject.toml)
uv run main.py

# in case you find the .venv too big and will leave this project for a while, you can remove it.
# as long as the cache is not cleared, it's very fast to rebuild the .venv.
rm -rf .venv
```