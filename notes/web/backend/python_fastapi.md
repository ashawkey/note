# fastapi

### install

```bash
# install fastapi and uvicorn
pip install fastapi[all]
```


### basic usage

```python
from fastapi import FastAPI

app = FastAPI()

### GET method for /
@app.get("/")
async def root():
    return {"message": "Hello World!"}

# also for POST, PUT, DELETE, ...
@app.post("/")
def rootpost():
    pass

# parameters
@app.get("/items/{id}")
def readitem(id: int):
    return {"item_id": id}

# order matters
# /who/me will invoke getme() since it is defined prior to getuser().
@app.get("/who/me")
def getme():
    pass

@app.get("/who/{user}")
def getuser(user):
    pass

### path parameters (e.g., pass in "/path/to/sth")
# /files//path/to/sth will get /path/to/sth
@app.get("/files/{file_path:path}")
def getfile(file_path):
    pass

### lookup parameters (e.g., ?key=val&key2=val2)
# /items/?index=10 will get index = 10
# /items/ will get index = 0, since we provide a default value, it is not a required parameter.
@app.get("/items/")
def getitem(index: int=0):
    return items[index]

# /items/ will throw an error now, since we provide no default, it is a required parameter.
@app.get("/items/")
def getitem(index: int):
    return items[index]

### post by pydantic
from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

@app.post("/post_item/")
def postitem(item: Item): # will decode the post body into a Item instance.
    item_dict = item.dict()
	return item_dict

# also support all the parameters
@app.post("/post_item/{id}")
def postitem(item: Item, id: int, q: int=0):
    item_dict = item.dict()
    item_dict['id'] = id
    item_dict['q'] = q
	return item_dict

### restricted parameters
# the passed in parameter must follow the defined rules.
from fastapi import Query, Path

# path (id, always required) + optional query (key1) + required query (key2)
@app.get("/items/{id}")
def getitem(
    id: int = Path(..., title="item id", gt=0, le=100),
    key1: str = Query("default", min_length=1, max_length=50, regex="^pattern$"),
    key2: str = Query(..., min_length=1, max_length=50, regex="^pattern$")
):
    pass


### error handling
from fastapi import HTTPException

@app.get("/items/{id}")
def getitem(id: int):
    if id not in ITEMS:
        raise HTTPException(status_code=404, detail="item not found")
    return {"item": ITEMS[id]}
```


### debug

the `main.py`:

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
	return {"message": "Hello World!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then run with:

```bash
python main.py
```


### deploy

assume the APIs are in `main.py`, and the backend is called `app`:

```bash
# dev example, will run at localhost:8000, and reload once source file changes.
uvicorn main:app --reload
```

It will serves:

* `localhost:8000/`: the api root.

* `localhost:8000/docs`: the api documentation.
* `localhost:8000/redoc`: another api documentation, better use with typing.
* `localhost:8000/openapi.json`: OpenAPI description.

More options:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

(However, logging with uvicorn seems to be painful.)