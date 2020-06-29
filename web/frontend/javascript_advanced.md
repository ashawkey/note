# JavaScript Advanced

### Strict mode

```javascript
"use strict";

function foo(){
    "use strict";
}
```



### Prototype Chain

* All the reference types are Objects.
* All the reference types have `_proto_` attribute.
* Function has `prototype` attribute.
* All the reference types' `_proto_` points to its constructor's `prototype`
* If an attribute cannot be found in an object, then find it in the object's `_proto_`.



### Hoisting

**Only hoists Declarations, not Initializations.**

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





