# Node.js

### `node`

Node.js is a **JavaScript run-time environment.**

It allows developers to write JavaScript outside of a browser.

| Difference |        |         |
| ---------- | ------ | ------- |
| Node.js    | no DOM | require |
| Browser    | DOM    | import  |

* CMD

  ```bash
  node app.js
  node app.js 1 2 3
  ```

  ```javascript
  // retrieve arguments
  process.argv.forEach((val, index) => {console.log(`${index}: ${val}`)})
  ```

  

* REPL

  ```bash
  node
  ```

  Special Commands:

  * `.help`
  * `.clear`
  * `.load <file>` 
  * `.save <file>`
  * `.exit` the REPL.

* Core Module: `process`

  ```javascript
  process.exit()
  process.env.ENV_VAR
  ```

  

* Input from CMD

  ```javascript
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  })
  
  readline.question(`What's your name?`, 
    name => {
      console.log(`Hi ${name}!`);
      readline.close();
    }
  )
  ```

* Expose

  ```javascript
  // car.js
  const car = {}
  module.exports = car
  // in other files:
  const car = require('./car.js')
  ```

  ```javascript
  // car.js
  exports.car = {}
  // in other files:
  const items = require('./car.js')
  items.car
  ```

  



### `npm`: Node Package Manager

```bash
npm install # based on package.json, install in local node_modules folder.
npm install <pkg> # specific, install in local folder, and add it to package.json.
npm install <pkg>@version # install old versions

npm install -g <pkg> # install in global folder.
npm install -S <pkg> # --save, also add ref in package.json
npm install -D <pkg> # --save-dev, also add dev-dependencies in package.json
npm root -g # the global folder. (/usr/local/lib/node_modules)

npm uninstall <pkg> 
npm uninstall -S <pkg>  # --save, also remove ref in package.json
npm uninstall -D <pkg>  # --save-dev, also remove dev-dependencies in package.json
npm uninstall -g <pkg>  # global

npm update # all
npm update <pkg> # specific

npm list # list all installed packages with version.
npm list <pkg> # list <pkg> with version.
npm list -g --depth 0
```

```json
// package.json
{
  "scripts": {
    "start-dev": "node lib/server-development",
    "start": "node lib/server-production"
  },
}
```

```bash
npm run start-dev # based on scripts in package.json
npm run start
```



### `npx`: Node Package Runner

```bash
npm install cowsay # executables in .bin
# how to run
./node_modules/.bin/cowsay
# a better solution
npx cowsay
```

```bash
npx node@10 # run specific version of node
npx https://xxx # run code from web
```



### `package.json`

```json
{
  "name": "test-project",
  "version": "1.0.0",
  "description": "A Vue.js project",
  "main": "src/main.js",
  "private": true,
  "scripts": {
    "dev": "webpack-dev-server --inline --progress --config build/webpack.dev.conf.js",
    "start": "npm run dev",
    "unit": "jest --config test/unit/jest.conf.js --coverage",
    "test": "npm run unit",
    "lint": "eslint --ext .js,.vue src test/unit",
    "build": "node build/build.js"
  },
  // Production
  "dependencies": {
    "vue": "^2.5.2"
  },
  // Development
  "devDependencies": {
    "autoprefixer": "^7.1.2",
    "babel-core": "^6.22.1",
  },
  "engines": {
    "node": ">= 6.0.0",
    "npm": ">= 3.0.0"
  },
  "browserslist": ["> 1%", "last 2 versions", "not ie <= 8"]
}
```

`package-lock.json`: the exact environment to reproduce the code.

