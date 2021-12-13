# Regex

```python
import re
```

### Raw string

```python
r'\s*'
```



### Special Characters

* `.`
* `^`, `$`
* `{m}`
* `{m, n}`,`{m, n}?`
* `*`, `*?`: {0, INF}
* `+`, `+?`: {1, INF}
* `?`, `??`: {0, 1}

* `\`
* `[]`, `[^]`: character set, auto-escape special chars.
* `|`
* `()`: group, use `\number` to catch (start from 1)
* `(?#...)`: comment
* `(?=...)`: lookahead assertion
* `(?!...)`: negative lookahead assertion

* `(?<=...)`: lookbehind assertion
* `(?<!...)`: negative lookbehind assertion

* `\s`: [ \t\n\r\f\v]
* `\S`: [ ^ \t\n\r\f\v]

* `\w`: [a-zA-Z0-9_]

* `\W`: [ ^a-zA-Z0-9_]



### API

```python
# search, anywhere
re.search(pattern, string) -> Match-Obj/None
# match, search from start
re.match(pattern, string) -> Match-Obj/None
# fullmatch, search from start to end (full)
re.fullmatch(pattern, string) -> Match-Obj/None
# split
re.split(pattern, string, maxsplit=0) -> list
# findall, finditer
re.findall(pattern, string) -> tuple
# sub
re.sub(pattern, repl, string, count=0) -> string
# escape
re.escape(string) -> string
# compile -> Pattern Object
prog = re.compile(pattern)
res = prog.match(string)
# Match Object
m.groups()
m.group(i)
m.pos
m.endpos
```



### Examples

```python
## extract content inside parenthesis
re.findall('\((.*?)\)', 'a(b), c (de), f(g(h))')
#['b', 'de', 'g(h']

## split by any space
re.split('\s*', 'a b    c\t d')
#['a', 'b', 'c', 'd']
```

