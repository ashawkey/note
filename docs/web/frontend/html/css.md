# CSS

> To make your website beautiful, we still need to write it.

```css
/* comment */
p {
    color: orange;
}
```

### Usage

* External

  ```html
  <link rel="stylesheet" href="styles/style.css">
  ```

* Internal

  ```html
  <style>
      p { color: orange; }
  </style>
  ```

### Selector

```css
p {} /* <p> */
p, li {}  /*<p> and <li> */
li em {} /* all <em> inside <li> */
li > em {} /* direct <em> son of <li> */
h1 + p {} /* the first {p} after <h1> */

 /* class='special' */
.special {}
li.special {} /* <li class='special'> */
    
/* id='index' */
#index {}

/* pseudo-class (state) */
a:link {color: pink;}
a:visited {color: green;}
a:hover {text-decoration: none;}

/* pseudo-element */
p::first_line {}

/* attribute */
a[attr] {} /* a with attr */
a[attr=value] {} /* a with attr = value */
a[attr^=value] {} /* a with attr starts with value */
a[attr$=value] {} /* a with attr ends with value */
a[attr*=value] {} /* a with value in attr */

/* all */
* {}
```

### Function

```css
.box {
  padding: 10px;
  width: calc(90% - 30px);
  background-color: rebeccapurple;
  color: white;
  transform: rotate(0.8turn)
}
```

### @ Rules

```css
/* import from another file */
@import 'styles2.css'; 


```


### Cascading and Inheriting

* Cascading

  ```css
  p {color: blue;}
  p {color: red;}
  
  /* then <p> is red */
  ```

* Priority

  ```html
  <h1 class="main-heading">This is my heading.</h1>
  ```

  More specific, higher priority.

  ```css
  .main-heading { 
      color: red; 
  }
          
  h1 { 
      color: blue; 
  }
  
  /* <h1> is red */
  ```

* Inherit

  ```html
  <p>We can change the color by targetting the element with a selector, such as this <span>span</span>.</p>
  ```

  ```css
  body {
      color: blue;
  }
  
  /* span is also blue. */
  ```

  ```html
  <ul>
      <li>Default <a href="#">link</a> color</li>                             // blue
      <li class="my-class-1">Inherit the <a href="#">link</a> color</li>      // green
      <li class="my-class-2">Reset the <a href="#">link</a> color</li>        // black
      <li class="my-class-3">Unset the <a href="#">link</a> color</li>        // green
  </ul>
  ```

  ```css
  body {
      color: green;
  }
            
  .my-class-1 a {
      color: inherit; /* inherit body */
  }
            
  .my-class-2 a {
      color: initial; /* the same with default */
  }
            
  .my-class-3 a {
      color: unset; /* return to default */
  }
  ```

* override

  sometimes it is just useful.

  ```css
  code {
      background-color: #ffffff !important;
  }
  ```

  
### References

* `background`

  ```css
  background: red url(bg-graphic.png) 10px 10px repeat-x fixed;
  /* equals to */ 
  background-color: red;
  background-image: url(bg-graphic.png);
  background-position: 10px 10px;
  background-repeat: repeat-x;
  background-scroll: fixed;
  ```

  
