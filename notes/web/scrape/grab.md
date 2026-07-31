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
selectorList = g.doc('/xpath') # g.doc.select('/xpath')
selector = selectorList[0]

selector.html()
selector.text()
```


### Example

crawl videos in `yhdm.so` .

```python
def crawl(keyword):
    urls = []
    g = Grab()
    g.go(f'http://www.yhdm.so/search/{keyword}/')
    candidate_selectors = g.doc(f'//div[@class="lpic"]/ul/li/a/@href')
    for candidate in candidate_selectors:
        g.go(f'http://www.yhdm.so{candidate.text()}')
        episode_selectors = g.doc('//div[@class="movurl"]/ul/li/a/@href')
        for episode in episode_selectors:
            g.go(f'http://www.yhdm.so{episode.text()}')
            title_selectors = g.doc('//div[@class="gohome l"]/h1')
            title = title_selectors[0].text()
            data_selectors = g.doc('//div[@id="playbox"]/@data-vid')
            url = data_selectors[0].text()
            if url[-4:] == "$mp4":
                url = url[:-4]
                print(f'[INFO] crawled {title} {url}')
                urls.append({
                    'formattedUrl': url,
                    'title': title,
                    'snipped': '',
                })
	return urls

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


