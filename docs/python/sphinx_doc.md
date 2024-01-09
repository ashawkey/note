## Sphinx

### install

```bash
pip install sphinx
```


### Layout

Run this at the root dir of your lib:

```bash
sphinx-quickstart docs
# separate source and build --> usually choose yes.
```

The layout will be:

```bash
mylib # <-- source code
setup.py
readme.md

# create by sphinx
docs
├── build # rendered doc
├── make.bat
├── Makefile
└── source
   ├── conf.py # important!
   ├── index.rst # welcome page
   ├── _static
   └── _templates
```

Build the doc by:

```bash
sphinx-build -M html docs/source/ docs/build

# or in linux
cd docs
make html
```

Then you can open the built index html (e.g., right click and openning by vs code live server):

http://127.0.0.1:5500/docs/build/html/index.html


### Write docs

By default sphinx uses **ReStructuredText (.rst)** mark up language, which is less common. But we can use `m2r2` extension to convert markdown (.md) to rst formats:

```python
  extensions = [
    "m2r2",
]

source_suffix = [".rst", ".md"]
```


#### Narrative documentation

Just create a new file under `docs/source`, like `index.md`.

Remember to cross-reference it to `index.md` under `toctree`:

```rst
.. table of content trees

.. toctree::
   :caption: Tutorials
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./mesh.md
   ./camera.md

.. toctree::
   :caption: API
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./utils.md
   ./cli.md
```


#### Automatic documentation

To automatically document all APIs, use these extensions:

```python
extensions = [
    'sphinx.ext.autodoc', # extract doc for single func/class
    'sphinx.ext.autosummary', # extract doc for a file
]
```

These extensions will extract docstring from `py` source files. 

There are different docstring styles, by default sphinx use a format as:

```python
def func(path):
    """
    Func description.
    
    :param path: The path of the file to wrap
    :type path: str
    :returns: A buffered writable file descriptor
    :rtype: BufferedFileStorage
    """
    ...
```

Another popular format is Google Python style:

```python
def func(arg1: int, arg2: str) --> bool:
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    return True
```

You need this extension to use it:

```python
extensions = ['sphinx.ext.napoleon']
```

To auto-generate the docs, you need to write in your doc file:

```rst
.. autodoc a func
.. autofunction:: kiui.func

.. autodoc a class
.. autoclass:: kiui.myClass()

.. autodoc a file (autosummary)
.. autosummary::
	:toctree: generated
	kiui
```

Note that `autodoc` will import the file to doc it! Ensure there are no side effects for any import.


### Publish 

Use extension:

```python
extensions = [
     "sphinx.ext.githubpages",
]
```


Create a doc building `docs/requirements.txt`:

```
sphinx
furo
m2r2
```


TODO


### Theme

For example:

```bash
pip install furo
```

Change conf.py:

```python
html_theme = 'furo'
```


### Example conf

```python
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'kiui'
copyright = '2024, kiui'
author = 'kiui'

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = "kiuikit"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

sphinx_to_github = True
sphinx_to_github_verbose = True
sphinx_to_github_encoding = "utf-8"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
```

