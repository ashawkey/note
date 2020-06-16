# JavaScript Basics

### Commons

```javascript
var a = 1; // use ;
/* this is a comment */
```



### Types

* `Number`

  * Double-precision 64-bit. **Only Float, No Integer.**

  * Convert

    ```javascript
    parseInt('100', 10); // =100
    parseInt('11', 2); // =3
    +'42'; // unary operator, =42
    parseInt('Hello', 10); // =NaN, isNaN()
    1/0; // =Infinity, isFinite()
    ```

  > `BigInt`: 
  >
  > ```javascript
  > const a = 99999999999999999999999999n
  > const b = BigInt(123456789123456789)
  > ```

  

* `String`

  * `utf-16`

  * Methods

    ```javascript
    'hello'.length; // 5
    'hello'.charAt(0); // 'h'
    'hello, world'.replace('world', 'mars'); // "hello, mars"
    ```

    

* `Boolean`

   * `true`
   * `false`: `0, '', "", NaN, null, undefined`

* `Object`
   Function, Array, Date, RegExp
   
* `Symbol`

* `null`: non-value

* `undefined`: uninitialized variable



### Variables

Pass by value.

* `var`

  **variables**. most common.

  Maybe Global or Local, depending on where we declare it.

* `let`

  **block-level variables**. 

  Visible in the block it is enclosed in.

  No hoisting. 

  Can be declared only once.

  ```javascript
  let a; // undefined
  let b = 1;
  for (let i=0; i<5; i++) {}
  
  {var i = 9;} 
  console.log(i);  // 9
  
  {let j = 9;} 
  console.log(j);  // Uncaught ReferenceError: j is not defined
  ```

* `const`

  **constants**.

  Maybe Global or Local, depending on where we declare it.



### Operators

```javascript
'3' + 4 + 5;  // "345"
 3 + 4 + '5'; // "75"

// ==, != do type conversion
123 == '123'; // true
1 == true; // true

// ===, !== avoid type conversion
123 === '123'; // false
1 === true;    // false
```



### Controls

```javascript
for (let value of array) {}
for (let prop in object) {}

switch (action) {
  case 'draw':
    drawIt();
    break;
  case 'eat':
    eatIt();
    break;
  default:
    doNothing();
}
```



### Objects

Objects in JavaScript are like Dictionaries in Python.

Pass by Reference.

* Create

  ```javascript
  var obj = new Object();
  var obj = {}; // object literal syntax
  var obj = {
    name: 'Carrot',
    details: {
      color: 'orange',
      size: 12
    }
  };
  ```

* Attribute Access

  ```javascript
  obj.details.color;
  obj['details']['color'];
  ```



### Arrays

A special type of Object. (Not a list, still a dictionary!)

```javascript
var a = new Array();
a[0] = 0;
a.length; // 1

var a = ['dog', 'cat', 'hen'];
a[100] = 'fox';
a.length; // 101
typeof a[90]; // undefined

a.push('fox');
a.push('a', 'b', 'c');
a.pop();
a.slice(start[, end]);
a.sort([cmpfn])
```



### Functions

* If no `return`, return `undefined`.

* Parameters

  ```javascript
  function add(x, y){
      var total = x + y;
      return total;
  }
  
  add(); // NaN
  add(1,2,4); // 3 (4 is ignored.)
  ```

* Arguments

  ```javascript
  function add() {
    var sum = 0;
    for (var i = 0, j = arguments.length; i < j; i++) {
      sum += arguments[i];
    }
    return sum;
  }
  add(2, 3, 4, 5); // 14
  
  function avg(...args) {
    var sum = 0;
    for (let value of args) {
      sum += value;
    }
    return sum / args.length;
  }
  avg(2, 3, 4, 5); // 3.5
  
  var avg = function() {
    var sum = 0;
    for (var i = 0, j = arguments.length; i < j; i++) {
      sum += arguments[i];
    }
    return sum / arguments.length;
  };
  ```

* Lambda function

  ```javascript
  var a = 1;
  var b = 2;
  
  (function() {
    var b = 3;
    a += b;
  })();
  
  a; // 4
  b; // 2
  ```

* IIFE (Immediately Invoked Function Expressions)

  ```javascript
  var charsInBody = (function counter(elm) {
    var count = 0;
    for (var i = 0, child; child = elm.childNodes[i]; i++) {
      count += counter(child); // recursion
    }
    return count;
  })(document.body);
  ```

  

### OOP

* JavaScript use Function as Classes.

  ```javascript
  function makePerson(name, age){
      return {
          // Attributes
          name: name,
          age: age,
          // Methods
          get_name: function() {return this.name;},
          get_age: function() {return this.age;},
      };
  }
  
  var p = makePerson('Tom', 23);
  p.get_name();
  ```

* `this`

  * Refers to

    * Function: current object.
    * Dot/Bracket notation: the leading object.
    * Else: the global object.

  * `Function.prototype.bind()`

    Bind `this` to an object.

    ```javascript
    var p = makePerson('Tom', 23);
    
    var get_name = p.get_name;
    get_name(); // undefined, this = the global object
    
    var get_name_p = get_name.bind(p); // set this = p
    get_name_p(); // 'Tom'
    ```

  * Use `this` to define a class

    ```javascript
    function Person(name, age) {
    	this.name = name;
        this.age = age;
        this.get_name = function() { return this.name; }
        this.get_age = function() { return this.age; }
    } // remember function is an object too.
    
    var p = new Person('Tom', 23)
    ```

    

  * `new`

    It creates a new empty object, then call the function specified, with `this` set to that new object.

    The function returns `undefined`, it's `new` that returns `this` object to the variable.

  

* `prototype`

  `X.prototype` is an object shared by all instances of `X`.

  ```javascript
  function Person(name, age) {
  	this.name = name;
      this.age = age;
  }
  
  Person.prototype.get_name = function() { return this.name; }
  Person.prototype.get_age = function() { return this.age; }
  ```

  We can use this to add methods to existing classes at any time!

  ```javascript
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

  

* `apply` and `call`

  ```javascript
  function trivialNew(constructor, ...args) {
    var o = {}; // Create an object
    constructor.apply(o, args); // apply(this, args)
    return o;
  }
  ```

  ```javascript
  function get_name_upper() {return this.name.toUpperCase();}
  var s = new Person('Tom', 23);
  get_name_upper.call(s);
  // Is the same as:
  s.get_name_upper = get_name_upper;
  s.get_name_upper(); // 'TOM'
  ```

* Inheritence

  ```javascript
  function Teacher(name, age, subject) {
      Person.call(this, name, age);
      this.subject = subject;
  }
  
  Teacher.prototype = Object.create(Person.prototype)
  ```

  

### Closures

```javascript
function makeAdder(a) {
  return function(b) {
    return a + b;
  };
}
var add5 = makeAdder(5);
var add20 = makeAdder(20);
add5(6); // 11
add20(7); // 27
```

