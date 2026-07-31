# Manifest

Assume file directory as:

```
# develop
- public
	- static
		- manifest.json
	- index.html
- src


# after build
- static
	- manifest.json
- index.html
```


```

```

This file determines a web app 's outlook.

```json
{
  "short_name": "nonsense", // name of app
  "name": "nonsense frontend",
  "icons": [
    {
      "src": "favicon.ico", // icon of app
      "sizes": "512x512",
      "type": "image/x-icon"
    }
  ],
  "start_url": "../index.html", // start url (relative to manifest.json)
  "display": "standalone",
  "theme_color": "#000000",
  "background_color": "#ffffff"
}

```

