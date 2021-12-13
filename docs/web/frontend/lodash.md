# lodash

A JS utility library imported as `_`.

Mainly for arrays, numbers, objects, strings.

### install

```bash
npm install -S lodash
```



### usage

```js
import _ f

_.defaults({ 'a': 1 }, { 'a': 3, 'b': 2 });
// → { 'a': 1, 'b': 2 }
_.partition([1, 2, 3, 4], n => n % 2);
// → [[1, 3], [2, 4]]
```

