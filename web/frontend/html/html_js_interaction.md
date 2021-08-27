```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title> example html file </title>
        
        <!-- import the src as plain text -->
        <script type="text/javascript" src="..."> </script>
        
    </head>
    <body>
        <button onclick="custom_func()" id="button"> Button </button>
        <!-- import the src as a module -->
        <script type="module" src="..."> </script>
    </body>
</html>
```



```js
function custom_func () {
    // use global object `document` to access the DOM
	let button = document.getElementById('button');
    button.disabled = true;
}

// to use it in html, bind it to global object `window`.
window.custom_func = custom_func;
```

