# pyodide

run python in javascript!

* Python runtime built into WebAssembly.
* Includes `NumPy, Pandas, Matplotlib, Scipy, scikit-learn`.
* transparent access between python and js.

### example in web browsers

```html
<!DOCTYPE html>
<html>
  <head>
      <script src="https://cdn.jsdelivr.net/pyodide/v0.18.1/full/pyodide.js"></script>
  </head>
  <body>
    Pyodide test page <br>
    Open your browser console to see Pyodide output
    <script type="text/javascript">
      async function main(){
        let pyodide = await loadPyodide({
          indexURL : "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/"
        });
        console.log(pyodide.runPython(`
            import sys
            sys.version
        `));
        console.log(pyodide.runPython("print(1 + 2)"));
      }
      main();
    </script>
  </body>
</html>
```


### example in nodejs

Install:

```bash
npm install pyodide
```

use:

```js
let pyodide_pkg = await import("pyodide/pyodide.js");

let pyodide = await pyodide_pkg.loadPyodide({
  indexURL: "<pyodide artifacts folder>",
});

await pyodide.runPythonAsync("1+1");
```


### JS API (`namedLikeThis`)

* `globals`: `PyProxy`, alias to the global python environment.

  ```js
  pyodide.globals.get("foo")
  ```

* `runPython(code, globals=pyodide.globals)`: run python code string.

* `runPythonAsync(code)`: async version, returns a Promise.

* `loadPackage(names)`: manually load python packages.

* `loadPackagesFromImports(code)`: automatically inspect the code string and load all packages needed.

* `toPy(obj, options)`: convert to `PyProxy`, but used in js.

  Most time we just use `JsProxy.to_py()` in python.

* `PyProxy`: A python object proxied to js.

  * `toJs(options)`: convert to a native js object.


### Python API (`named_like_this`)

* `js`: alias to the global js environment (`window`).
* `JsProxy`: A js object proxied to python.
  * `to_py(depth=-1)`: convert to a native python object. (deep conversion by default)

* `pyodide.to_js(obj, depth=-1)`: convert to `JsProxy`, but used in python.

  Most time we just use `PyProxy.toJs()` in js.


### Micropip API

* `install(requirements)`: install given packages and dependencies.


### access python from js

```js
pyodide.runPython(`
  import numpy
  x=numpy.ones((3, 4))
`);
pyodide.globals.get('x').toJs();
// >>> [ Float64Array(4), Float64Array(4), Float64Array(4) ]

// create the same 3x4 ndarray from js
x = pyodide.globals.get('numpy').ones(new Int32Array([3, 4])).toJs();
// x >>> [ Float64Array(4), Float64Array(4), Float64Array(4) ]

// re-assign a new value to an existing variable
pyodide.globals.set("x", 'x will be now string');

// create a new js function that will be available from Python
// this will show a browser alert if the function is called from Python
pyodide.globals.set("alert", alert);

// this new function will also be available in Python and will return the squared value.
pyodide.globals.set("square", x => x*x);

// You can test your new Python function in the console by running
pyodide.runPython("square(3)");

// avoid memory leak: always delete toJS objects:
let pyproxies = [];
proxy.toJs({pyproxies});
// Do stuff
for(let px of pyproxies){
    px.destory();
}
proxy.destroy();
```

np array to js:

```js
let proxy = pyodide.globals.get("some_numpy_ndarray");
let buffer = proxy.getBuffer();
proxy.destroy();
try {
  if (buffer.readonly) {
    // We can't stop you from changing a readonly buffer, but it can cause undefined behavior.
    throw new Error("Uh-oh, we were planning to change the buffer");
  }
  let array = new ndarray(
    buffer.data,
    buffer.shape,
    buffer.strides,
    buffer.offset
  );
  // manipulate array here
  // changes will be reflected in the Python ndarray!
} finally {
  buffer.release(); // Release the memory when we're done
}
```

python objects to js:

```js
let sys = pyodide.globals.get("sys");
```


### access js from python

```python
import js # js is `window` in browser

# directly modify web
div = js.document.createElement("div")
div.innerHTML = "<h1>This element was created from Python</h1>"
js.document.body.prepend(div)

# fetch in python
from js import fetch
resp = await fetch('example.com/some_api',
    method= "POST",
    body= '{ "some" : "json" }',
    credentials= "same-origin",
)

# some proxy methods
x = div
str(x) # x.toString()
len(x) # x.length or x.size
x.typeof # typeof x
await x # await x
```

js arrays:

```js
self.jsarray = new Float32Array([1,2,3, 4, 5, 6]);
pyodide.runPython(`
    from js import jsarray
    array = jsarray.to_py()
    import numpy as np
    numpy_array = np.asarray(array).reshape((2,3))
    print(numpy_array)
`);
```

js object to python:

```python
import js
js.document.title = 'New window title'
from js.document.location import reload as reload_page
reload_page()
```


### load python packages

```js
// load official supported packages
await pyodide.loadPackage("numpy");
await pyodide.loadPackage(["cycler", "pytz"]); // multiple at one time


// install from pypi through micropip
await pyodide.runPythonAsync(`
  import micropip
  await micropip.install('snowballstemmer')
  import snowballstemmer
  stemmer = snowballstemmer.stemmer('english')
  print(stemmer.stemWords('go goes going gone'.split()))
`);

// install wheels from url
await pyodide.runPythonAsync(`
  import micropip
  micropip.install('https://example.com/files/snowballstemmer-2.0.0-py2.py3-none-any.whl')
`);
```

