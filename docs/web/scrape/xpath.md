# XPath

Cheat sheet: https://devhints.io/xpath

### Examples

```xquery
# target                x
div p	                //div//p
ul > li	                //ul/li
a:first-child	        //a[1]
ul > li:last-child	    //ul/li[last()]

#id	                    //*[@id="id"]	
.class	                //*[@class="class"]
input[type="submit"]	//input[@type="submit"]	 
a#abc[for="xyz"]	    //a[@id="abc"][@for="xyz"]	
a[rel]	                //a[@rel]	 
a[href^='/']	        //a[starts-with(@href, '/')]	
a[href*='://']	        //a[contains(@href, '://')]	 

href value              //a/@href

Text match	            //button[text()="Submit"]
Has children	        //ul[*]
Has children (specific)	//ul[li]

logical                 //div[@id="head" and position()=2]
count                   //ul[count(li) > 2]
```



### Selectors

```
name
/div
div/p
//div
.
..
@property
A | B
```

```
/div[1]
/div[last()]
/div[position()<3]
//title[@lang='eng']
```

```
*
@*
node()
```



### Built-in functions

```
name() # h1, h2, div, ...
text() # plain text
count(div), count(//*) 
position()
not(expr)
contains(@class, 'head')
starts-with(@class, 'head')
ends-with(@class, 'head')
concat(a, b)
substring(str, start, end)

```



### Operators

```
|
+ - * div 
= !=
< <= > >=
or and
mod
```



### Axis

```
[default] = child::
@ = attribute::
// = /descendant-or-self::
. = self::node()
.. = parent::node()
```

