# HTML

> We prefer not to write HTML directly. Instead, use code to generate HTML automatically.



### Elements

* `<a href='https://example.com'> Website </a>`
* `<b> Bold </b>`: bold
* `<i> Italic </i>`: Italic
* `<em> Emphasize </em>`: Usually Italic.
* `<br>`: line break

* `<link href='style.css' rel='stylesheet'>`: link a external resource.
  * `href`: URL of the resource.
  * `rel`: relationship to the resource. 

* `<canvas>`
* `<span>` inline element
* `<div>` block element

* `<input>`

  https://developer.mozilla.org/zh-CN/docs/Web/HTML/Element/Input

  ```html
  <input type="text" />
  ```

* `<label>`
* `<form>`

* `<header>` declare header, no factual effect.







### Example

* Simple Document

  ```html
  <!DOCTYPE html>
  <html>
  <body>
  
  <h1 style="text-align:center;">Centered Heading</h1>
  <p style="text-align:center;">Author</p>
  <hr/>
  <p> A long long story.</p>
  
  
  </body>
  </html>
  ```

  



### Script

basics:

```html
<!-- include a source file -->
<script src="file.js"></script>

<!-- directly write in html -->
<script>
    console.log('Hello');
</script>

<!-- type="module" allow us to use "import" -->
<script type="module">
    import * as THREE form 'three'
</script>
```



