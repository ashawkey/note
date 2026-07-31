# asynchronous programming


### Promise

A `Promise`  is an object representing the eventual completion or failure of an asynchronous operation.

e.g., `fetch()` returns a `Promise`.

It is always in one of these states: `pending, fufilled, rejected.`

Use `then()` to add callbacks to `fufilled` promises, and `catch()` for `rejected` promises.


Some guarantees:

* Callbacks added with `then()/catch()` will never be invoked before the [completion of the current run](https://developer.mozilla.org/en-US/docs/Web/JavaScript/EventLoop#run-to-completion) of the JavaScript event loop
* These callbacks will be invoked even if they were added *after* the success or failure of the asynchronous operation that the promise represents.
* (Chaining) Multiple callbacks may be added by calling `then()` several times. They will be invoked one after another, in the order in which they were inserted.


Chaining:

```js
doSomething()
.then(result => doSomethingElse(result))
.then(newResult => doThirdThing(newResult))
.then(finalResult => {console.log(`Got the final result: ${finalResult}`);})
.catch(failureCallback);

doAnotherthing(); // this will not wait until doSomething() finish !
```


Catching errors:

```js
new Promise((resolve, reject) => {
    console.log('Initial');
    resolve();
})
.then(() => {
    throw new Error('Something failed');
    console.log('Do this');
})
.catch(() => {
    console.error('Do that');
})
.then(() => {
    console.log('Do this, no matter what happened before');
});
```


### Create promises manually

```js
let p = new Promise((resolveFunc[, rejectFunc]) => {
	// do something
    // if you returned something, it will be passed to `then`.
});

p.then(handleResolveFunc[, handleRejectFunc]);

Promise.resolve([value]) // returns a dummy resolved Promise with value.
```

examples:

```js
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

wait(10*1000).then(() => console.log("10 seconds later"));
```

```js
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

wait(0).then(() => console.log(4)); // need to resolve (even 0 ms), second in task queue.
Promise.resolve().then(() => console.log(2)).then(() => console.log(3)); // already resolved promise, first in task queue
console.log(1); // not in task queue

// 1, 2, 3, 4
```


### async / await

`async` can be put in front of a function to make it async, i.e., return a Promise.

```js
async function hello() { return "Hello" };
hello(); // return a Promise
hello().then((value) => console.log(value)); // output Hello.
hello().then(console.log) // shorter ver.
```

`await` can be put in front of a `Promise` to **pause program until it fulfills**.

```js
async function hello() {
  return greeting = await Promise.resolve("Hello");
};

hello().then(alert);
```

They can be used to replace `then()` chains:

```js
// then ver.
fetch('coffee.jpg')
.then(response => {
  if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`);}
  return response.blob();
})
.then(myBlob => {
  let objectURL = URL.createObjectURL(myBlob);
  let image = document.createElement('img');
  image.src = objectURL;
  document.body.appendChild(image);
})
.catch(e => {
  console.log('There has been a problem with your fetch operation: ' + e.message);
});

// async ver.
async function myFetch() {
  let response = await fetch('coffee.jpg');
  if (!response.ok) {throw new Error(`HTTP error! status: ${response.status}`);}
  let myBlob = await response.blob();
  let objectURL = URL.createObjectURL(myBlob);
  let image = document.createElement('img');
  image.src = objectURL;
  document.body.appendChild(image);
}

myFetch()
.catch(e => {
  console.log('There has been a problem with your fetch operation: ' + e.message);
});
```

