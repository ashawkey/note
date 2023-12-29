# Web APIs

### DOM (Document Object Model)

DOM connects Web pages to scripts by representing the structure of a document.

```javascript
// Window 
window.document
window.onload

// Document 
document.getElementById(id)
document.getElementByTagName(name)
document.createElement(name)

// Element
element.getAttribute()
element.setAttribute()

```

### Storage

`localStorage` maintains a separate storage space for each given origin.

It persists even when browser is closed and reopened. (better than cookies)

```javascript
localStorage.colorSetting = '#a4509b';
localStorage.setItem('colorSetting', '#a4509b');
let cat = localStorage.getItem('myCat');

function storageAvailable(type) {
    var storage;
    try {
        storage = window[type];
        var x = '__storage_test__';
        storage.setItem(x, x);
        storage.removeItem(x);
        return true;
    }
    catch(e) {
        return e instanceof DOMException && (
            // everything except Firefox
            e.code === 22 ||
            // Firefox
            e.code === 1014 ||
            // test name field too, because code might not be present
            // everything except Firefox
            e.name === 'QuotaExceededError' ||
            // Firefox
            e.name === 'NS_ERROR_DOM_QUOTA_REACHED') &&
            // acknowledge QuotaExceededError only if there's something already stored
            (storage && storage.length !== 0);
    }
}

if (storageAvailable('localStorage')) {
  // Yippee! We can use localStorage awesomeness
}
else {
  // Too bad, no localStorage for us
}
```


### URL Parameters

`https://example.com/?product=shirt&color=blue&newuser&size=m`

Here the URL parameters are:

```
product=shirt
color=blue
newuser
size=m
```

This allows to pass simple data to the server in `GET` and `POST`.


### Fetch

```javascript
// Promise<Response> fetch(input[, init]);

// get
fetch('http://example.com/movies.json')
.then(res => res.json()) // necessary, Promise -> json
.then(response => console.log(response));

// post
var url = 'https://example.com/profile';
var data = {username: 'example'};

fetch(url, {
  method: 'POST',
  body: JSON.stringify(data), // data can be `string` or {object}!
  headers: new Headers({
    'Content-Type': 'application/json' // data type
  })
}).then(res => res.json())
.catch(error => console.error('Error:', error))
.then(response => console.log('Success:', response));
```

Fetch with URL query parameters (https://github.com/github/fetch/issues/256):

```javascript
var params = {
    parameter1: 'value_1',
    parameter2: 'value 2',
    parameter3: 'value&3' 
};

var query = Object.keys(params)
    .map(k => encodeURIComponent(k)+'='+encodeURIComponent(params[k]))
    .join('&');

fetch(url+"?"+query).then(...)
```


