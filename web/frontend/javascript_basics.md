# JavaScript Basics

### Commons

```javascript
var a = 1; // use ;
/* this is a comment */
{} // this is an object. (it works like python dict)
```



### Types

* `Number`

  * Double-precision 64-bit. **Only Float, No Integer.**

  * Convert

    ```javascript
    100.toString()
    
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

  * template strings

    ```js
    "" // normal strings 1
    '' // normal strings 2
    `` // template strings
    
    `\`` === '`' // --> true
    `string text ${expression} string text` // support variables!
    `line1
     line2` // support multi-line
    
    // tagged templates. An advanced string manipulation way.
    let person = 'Mike';
    let age = 28;
    
    function myTag(strings, personExp, ageExp) {
      let str0 = strings[0]; // "That "
      let str1 = strings[1]; // " is a "
      let str2 = strings[2]; // "."
    
      let ageStr;
      if (ageExp > 99){
        ageStr = 'centenarian';
      } else {
        ageStr = 'youngster';
      }
    
      // We can even return a string built using a template literal
      return `${str0}${personExp}${str1}${ageStr}${str2}`;
    }
    
    let output = myTag`That ${ person } is a ${ age }.`;
    
    console.log(output);
    // That Mike is a youngster.
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

  **variables**. most common. (however deprecated.)

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
for (let prop in object) {} // depracated

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

// splice(start, delete_num, append)
a.splice(5, 1) // delete index=5 element
a.splice(0, 0, 'x') // insert 'x' at index=0
a.splice(a.indexOf('x'), 1) // delete 'x'

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




### Exports & Imports

Export by:

```javascript
// App.js
export default App;
export {function1, variable1};
export function function2() {}
```

Import by:

```javascript
// default
import App from "./App"

// interfaces
import {function1, variable1} from "./App"

// mixed
import App, {function1 as foo, function2, variable1 as bar} from "./App"

// all
import * as myModule from "./App"
myModule.App

// only run global code, without importing anything.
import '/modules/my-module.js';

// dynamic
import(".App.js").then(
	(module) => {
    	...    
    }
);

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

