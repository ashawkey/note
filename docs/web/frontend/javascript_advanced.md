# JavaScript Advanced

### Strict mode

```javascript
"use strict";

function foo(){
    "use strict";
}
```



### [`globalThis`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/globalThis)

A standard way to access the global environment in different js environments. 

e.g., It equals to `window, self, frames` in browser, and `global` in nodejs.

```js
// it works like this:
var getGlobal = function () {
  if (typeof self !== 'undefined') { return self; }
  if (typeof window !== 'undefined') { return window; }
  if (typeof global !== 'undefined') { return global; }
  throw new Error('unable to locate global object');
};

var globals = getGlobal();
```



### Prototype Chain

* All the reference types are Objects.
* All the reference types have `_proto_` attribute.
* Function has `prototype` attribute.
* All the reference types' `_proto_` points to its constructor's `prototype`
* If an attribute cannot be found in an object, then find it in the object's `_proto_`.



### Hoisting (声明提前)

Function and Variable declarations will be hoisted at runtime (in their domains, i.e., global or local function). 

However, it **only hoists Declarations, not Initializations.**

```javascript
var a = 99;            // 全局变量a
f();                   // f是函数，虽然定义在调用的后面，但是函数声明会提升到作用域的顶部。 
console.log(a);        // a=>99,  此时是全局变量的a
function f() {
  console.log(a);      // 当前的a变量是下面变量a声明提升后，默认值undefined
  var a = 10;
  console.log(a);      // a => 10
}

// 实际运行顺序
var a = 99;
function f() {
  var a;
  console.log(a);     
  a = 10;
  console.log(a);
}
f();
console.log(a);

// 输出结果：
undefined
10
99
```



### Class (ES6)

##### basics

* Classes are "special functions".

  ```js
  // function syntax
  function Person(name, age){
      return {
          name: name,
          age: age,
          get_name: function() {return this.name;},
          get_age: function() {return this.age;},
      };
  }
  
  var p = Person('Tom', 23);
  p.get_name();
  
  // class syntax
  class Person {
      constructor(name, age) {
          this.name = name;
          this.age = age;
      }
      get_name() {return this.name;}
      get_age() {return this.age;}
  }
  
  var p = new Person('Tom', 23); // create a new empty object.
  p.get_name();
  ```

* `new`:

  It creates a new empty object, then call the function specified, with `this` set to that new object.

  The function returns `undefined`, it's `new` that returns `this` object to the variable.

* `prototype`:

  `X.prototype` is an object shared by all instances of `X`.

  ```js
  // prototype way to define a class
  function Person(name, age) {
  	this.name = name;
      this.age = age;
  }
  
  Person.prototype.get_name = function() { return this.name; }
  Person.prototype.get_age = function() { return this.age; }
  ```

  Use it to add flexibility to any class:

  ```js
  var s = 'Simon';
  s.reversed(); // TypeError on line 1: s.reversed is not a function
  
  String.prototype.reversed = function() {
    var r = '';
    for (var i = this.length - 1; i >= 0; i--) {
      r += this[i];
    }
    return r;
  };
  
  s.reversed(); // nomiS
  ```

  

* `this`:

  it can refers to different things under different contexts:

  * Function: current object.
  * Dot/Bracket notation: the leading object.
  * Else: the global object.

  A `this` way to build a class:

  ```js
  function Person(name, age) {
  	this.name = name;
      this.age = age;
      this.get_name = function() { return this.name; }
      this.get_age = function() { return this.age; }
  } // remember function is an object too.
  
  var p = new Person('Tom', 23)
  ```

  

  We can use `Function.prototype.bind()` to change `this`:

  ```js
  var p = makePerson('Tom', 23);
  
  var get_name = p.get_name;
  get_name(); // undefined, this = the global object
  
  var get_name_p = get_name.bind(p); // set this = p
  get_name_p(); // 'Tom'
  ```

  

* Class definitions **are not hoisted**. You must declare it first, then use it.
* Other examples:

```js
//// defeine a class
class Rectangle {
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }
}

let Rectangle = class {
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }
};

// tricky, the class name is in fact Rectangle2. use `Rectangle.name` to see it.
let Rectangle = class Rectangle2 {
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }
};

//// class body
// member variables and methods
class Rectangle {
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }
  // Getter
  get area() {
    return this.calcArea();
  }
  // Method
  calcArea() {
    return this.height * this.width;
  }
}

const square = new Rectangle(10, 10);
console.log(square.area); // 100

// static variables & methods (Acess by class name!)
class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  static displayName = "Point";
  static distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;

    return Math.hypot(dx, dy);
  }
}

const p1 = new Point(5, 5);
const p2 = new Point(10, 10);
p1.displayName; // undefined
p1.distance;    // undefined

console.log(Point.displayName);      // "Point"
console.log(Point.distance(p1, p2)); // 7.0710678118654755

//// public class fields (experimental)
class Rectangle {
  height = 0; // support default value
  width;
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }
}

//// private class fields
class Rectangle {
  #height = 0; // can only be accessed inside class body
  #width;
  constructor(height, width) {
    this.#height = height;
    this.#width = width;
  }
}

//// inheritance by `extends`
class Animal {
  constructor(name) {
    this.name = name;
  }
  speak() {
    console.log(`${this.name} makes a noise.`);
  }
}

class Dog extends Animal {
  constructor(name) {
    super(name); // call the super class constructor and pass in the name parameter
  }

  speak() {
    console.log(`${this.name} barks.`);
  }
}

let d = new Dog('Mitzie');
d.speak(); // Mitzie barks.

//// inheritance by prototype
function Teacher(name, age, subject) {
    Person.call(this, name, age);
    this.subject = subject;
}

Teacher.prototype = Object.create(Person.prototype)

//// instance of
console.log(d instanceof Dog); // true

//// super
class Cat {
  constructor(name) {
    this.name = name;
  }

  speak() {
    console.log(`${this.name} makes a noise.`);
  }
}

class Lion extends Cat {
  speak() {
    super.speak();
    console.log(`${this.name} roars.`);
  }
}

let l = new Lion('Fuzzy');
l.speak();
// Fuzzy makes a noise.
// Fuzzy roars.
```





### Tips

* `var` is deprecated. Use `const` and `let`.
* use `for (elem of collection)`, not `for (elem in collection)`
