# Grab

## Simple API



### Get

```python
from grab import Grab

g = Grab()
resp = g.go('https://openaccess.thecvf.com/CVPR2020?day=2020-06-16')
# g.doc == resp -> True
```

### Post

```python
from grab import Grab

g = Grab()

g.go('https://github.com/login')
g.doc.set_input('login', '****')
g.doc.set_input('password', '****')
g.doc.submit()

g.doc.save('/tmp/x.html')
home_url = g.doc('//a[contains(@class, "header-nav-link name")]/@href').text()
```

### Handle Response

```python
resp.url
resp.code

resp.body
resp.unicode_body()

resp.cookies
```

### Selector

```python
selectorList = g.doc('xpath') # g.doc.select('xpath')
selector = selectorList[0]

selector.html()
selector.text()
```





## Spider API

```python
from grab.spider import Spider, Task

class ExampleSpider(Spider):
    def task_generator(self):
        for lang in 'python', 'ruby', 'perl':
            url = 'https://www.google.com/search?q=%s' % lang
            yield Task('search', url=url, lang=lang)

    def task_search(self, grab, task):
        print('%s: %s' % (task.lang,
                          grab.doc('//div[@class="s"]//cite').text()))


bot = ExampleSpider(thread_number=2)
bot.run()
```



