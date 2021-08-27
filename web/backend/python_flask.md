# Flask (Pallets Project)

### CLI

```bash
## Set
export FLASK_APP=hello.py
# $env.FLASK_APP=hello.py

## Debug mode
export FLASK_ENV=development

## Run
flask run # python -m flask run
```



### Views

A function to respond to requests.

```python
@app.route("/hello")
def hello():
    return "Hello!"
```

Valid response:

* `string`

* `json`

  ```python
  from flask import jsonify
  def hello():
  	return jsonify({"time": time.time()})
  ```

* `render_template()`

  ```python
  def hello():
      return render_template("hello.html")
  ```

  

### Blueprint

A way to organize a group of views. (Modularization)

```python
from flask import Blueprint, request, redirect, url_for

bp = Blueprint('auth', __name__, url_prefix='/auth')
# register: app.register_blueprint(bp)

# add views
@bp.route('/register', methods=('GET', 'POST'))
def register():
    # request is imported
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        ...
        return redirect(...)
    #elif request.method == 'GET':
    return render_template(...)

@bp.route('/login', methods=('GET', 'POST'))
def login():
    pass

# other utils
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
```



### Context

**How flask handle request: **

* Create APP Context 
* Create Request Context 
* Do some Logic 
* Destroy Request Context 
* Destroy APP Context

**Difference:**

APP context is cheap, and can be created without a request, for accessing `current_app` and `g`.

Request context is expensive.



### URL arguments

* simple arguments

  ```js
  fetch("api/a/1");
  ```

  

  ```python
  @app.route("api/<arg1>/<arg2>")
  def func(arg1, args):
      return jsonify({
          'arg1': arg1,
          'arg2': arg2,
      })
  ```

* form

  ```javascript
  let formData = new FormData();
  formData.append('username', 'John');
  formData.append('password', '123');
  
  fetch("api/form",
      {
          method: "POST",
      	body: formData,
      });
  ```

  ```python
  @app.route("api/form", methods=('GET','POST'))
  def login():
      if request.method == 'POST':
  	    username = request.form.get('username')
      	password = request.form.get('password')
  ```

* URL parameters

  ```javascript
  fetch("login?username=alex&password=pw1")
  ```

  ```python
  @app.route("login", methods=("GET",))
  def login():
      username = request.args.get('username')
      password = request.args.get('password')
  ```

* `JSON`

  ```javascript
  fetch("test", {
      method: "POST",
      headers: {
          "Content-Type": "application/json",
      },
      body: JSON.stringify({name:"Alex"})
  })
  ```

  ```python
  @app.route("test", methods=('GET','POST'))
  def login():
      if request.method == 'POST':
  	    username = request.json.get('username')
      	password = request.json.get('password')
  ```

  

* `files`

  ```python 
  request.files
  ```

  

### flask.session

Store data **in different requests from the same client**. (signed cookies)

It is like a `dict`.

```python
session['user_id'] = user['id']

session.clear()
```



### flask.g

Store data **in the same request**.

It works like a `dict`.

```python
g.user = user

if g.user is not None:
    pass
```

Usual way to manage resource:

```python
def get_db():
    if 'db' not in g:
        g.db = connect_to_database()

    return g.db

@app.teardown_appcontext
def teardown_db():
    db = g.pop('db', None)

    if db is not None:
        db.close()
```





### Jinja

A template library.

- `{% ... %}` for [Statements](https://jinja.palletsprojects.com/en/2.11.x/templates/#list-of-control-structures)
- `{{ ... }}` for [Expressions](https://jinja.palletsprojects.com/en/2.11.x/templates/#expressions) to print to the template output
- `{# ... #}` for [Comments](https://jinja.palletsprojects.com/en/2.11.x/templates/#comments) not included in the template output
- `# ... ##` for [Line Statements](https://jinja.palletsprojects.com/en/2.11.x/templates/#line-statements)





