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

.. autodoc a class, including all members 
.. autoclass:: kiui.myClass()
   :members:
   
.. autodoc a file, including all functions and classes, most useful!
.. automodule:: kiui.utils
   :members:
```

Note that `autodoc` will import the file to doc it! Ensure there are no side effects for any import.

Note the **number of spaces** before `:members:`! It must be **2 or 3 spaces**, and 4 spaces will complain `Explicit markup ends without a blank line; unexpected unindent` and won't correctly generate your doc!


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
sphinx_design
sphinx-copybutton
furo
m2r2
```

Make sure the setup of your lib includes all necessary dependencies (may use a `[full]` dependency setting), since the sphinx need to import all modules to perform auto doc.

Create workflow under `.github/workflows/docs.yaml`:

```yaml
name: docs

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      # Check out source
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.10"

      # Build documentation
      - name: Building documentation
        run: |
          pip install --upgrade pip
          pip install -e ".[full]"
          pip install -r docs/requirements.txt
          sphinx-build docs/source docs/build -b dirhtml

      # Deploy
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }} # Note that the GITHUB_TOKEN is NOT a personal access token. A GitHub Actions runner automatically creates a GITHUB_TOKEN secret to authenticate in your workflow. So, you can start to deploy immediately without any configuration.
          publish_dir: ./docs/build
          cname: kit.kiui.moe
```

Push to your git repo and trigger the workflow. Don't forget to configure the github pages!


### Theme

For example:

```bash
pip install furo
```

Change conf.py:

```python
html_theme = 'furo'
```

You may also want to add a badge to github and show the stars, this will need to create a file under `source/_templates/sidebar/brand.html`:

```html
<a class="sidebar-brand{% if logo %} centered{% endif %}" href="{{ pathto(master_doc) }}">
    {% block brand_content %} {%- if logo_url %}
    <div class="sidebar-logo-container">
        <img class="sidebar-logo" src="{{ logo_url }}" alt="Logo" />
    </div>
    {%- endif %} {%- if theme_light_logo and theme_dark_logo %}
    <div class="sidebar-logo-container" style="margin: .5rem auto .5rem auto">
        <img class="sidebar-logo only-light" src="{{ pathto('_static/' + theme_light_logo, 1) }}" alt="Light Logo" />
        <img class="sidebar-logo only-dark" src="{{ pathto('_static/' + theme_dark_logo, 1) }}" alt="Dark Logo" />
    </div>
    {%- endif %} {#- {% if not theme_sidebar_hide_name %}
    <span class="sidebar-brand-text">{{ docstitle if docstitle else project }}</span>
    {%- endif %} -#} {% endblock brand_content %}
</a>

<div style="text-align: center">
    <b>kiuikit</b>
    </br>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
    <a class="github-button" href="https://github.com/ashawkey/kiuikit"
        data-color-scheme="no-preference: light; light: light; dark: light;" data-size="large" data-show-count="true"
        aria-label="Download buttons/github-buttons on GitHub">
        Github
    </a>
</div>
```

The logo and website favicon should be put under `source/_static/`, and set in `conf.py`:

```python
html_favicon = '_static/icon.png'
html_logo = '_static/logo.png'
```


### Example conf

```python
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
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "m2r2",
]

# sort automodule generated doc by source
autodoc_default_options = {
    "member-order": "bysource",
}

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
# html_title = "kiuikit" # default is name-version documentation

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_favicon = '_static/icon.png'
html_logo = '_static/logo.png'

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

