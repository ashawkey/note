# webpack

Another solution except for `react`.



### init project

```bash
npm init -y
npm install --save-dev webpack webpack-cli
```

this creates `package.json`

```json
 {
   "name": "webpack-demo",
   "version": "1.0.0",
   "description": "",
   "private": true, # avoid wrong publishing
   "scripts": {
     "test": "echo \"Error: no test specified\" && exit 1"
   },
   "keywords": [],
   "author": "",
   "license": "ISC",
   "devDependencies": {
     "webpack": "^5.4.0",
     "webpack-cli": "^4.2.0"
   }
 }
```



### files organization

we can install any package by `npm`:

```bash
npm install -S lodash
```



`webpack.config.js`

```js
const path = require('path');

module.exports = {
  // this generates main.js from index.js
  entry: './src/index.js',
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'dist'),
  },
};
```



`dist/index.html`

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Getting Started</title>
  </head>
  <body>
    <!-- include generated scripts manually -->
    <script src="main.js"></script>
  </body>
</html>
```



`src/index.js`

```js
// we can directly import it, instead of './node_modules/lodash'
import _ from 'lodash';

function component() {
  const element = document.createElement('div');

  element.innerHTML = _.join(['Hello', 'webpack'], ' ');

  return element;
}

document.body.appendChild(component());
```



### build

* directly use `npx`

  ```bash
  npx webpack # --config webpack.config.js
  ```

* Or write a snippet in `package.json`

  ```json
  {
      ...
      "scripts": {
          "build": "webpack"
      }
      ...
  }
  ```

  then

  ```bash
  npm run build
  ```

  