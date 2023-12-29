# HTTP

### Overview

HTTP is an Application-layer protocol to transmit hypermedia documents.

* Client-Server model.

  ![Client-Server model](https://mdn.mozillademos.org/files/13679/Client-server-chain.png)

* Stateless, but not sessionless (use cookies).

* HTTP Messages

  * Requests

    ```bash
    # Method Path Version
    GET / HTTP/1.1
    # Headers
    Host: developer.mozilla.org
    Cookie: yummy_cookie=choco; tasty_cookie=strawberry
    Accept-Language: fr
    ```

    ```bash
    POST /contact_form.php HTTP/1.1
    Host: developer.mozilla.org
    Content-Length: 64
    Content-Type: application/x-www-form-urlencoded
    
    name=Joe%20User&request=Send%20me%20one%20of%20your%20catalogue
    ```

    

  * Responses

    ```bash
    # Version Status_Code Statue_Message
    HTTP/1.1 200 OK
    # Headers
    Date: Sat, 09 Oct 2010 14:28:02 GMT
    Server: Apache
    Last-Modified: Tue, 01 Dec 2009 20:18:22 GMT
    ETag: "51142bc1-7449-479b075b2891b"
    Accept-Ranges: bytes
    Content-Length: 29769
    Content-Type: text/html
    Set-Cookie: yummy_cookie=choco
    Set-Cookie: tasty_cookie=strawberry
    
    <!DOCTYPE html... (here comes the 29769 bytes of the requested web page)
    ```

    

  

### Cache

A technique that stores a copy of a given resource, and serves it back when requested.

It eases the load of server, and improves performance.


### Cookies

A small piece of data that a server sends to the user's web browser.

The browser may store it, and send it back with later requests to the server.

Usually used to tell if two requests came from the same browser. (keeping logged-in)

* Expires

  ```bash
  Set-Cookie: id=a3fWa; Expires=Wed, 21 Oct 2015 07:28:00 GMT;
  ```

* Secure

  ```bash
  Set-Cookie: id=a3fwa; Secure; HttpOnly
  ```

  `HttpOnly` makes `Document.cookie` API cannot access this cookie.

* JavaScript API

  ```javascript
  // show cookies
  console.log(document.cookie); 
  
  // create new cookie
  document.cookie = "yummy_cookie=choco"; 
  document.cookie = "tasty_cookie=strawberry"; 

  // change cookie
  document.cookie = "tasty_cookie=mellon";   // overwrite
  
  // delete cookie
  document.cookie = "tasty_cookie=mellon; expires=Thu, 01 Jan 1970 00:00:00 UTC;"; // set expires to old time
  ```
  
  


### CORS: Cross-Origin Resource Sharing

A Mechanism that uses additional HTTP Headers, to tell browsers to give a web App access to resources from a different origin.

![](https://mdn.mozillademos.org/files/14295/CORS_principle.png)

CORS use headers to work.

Simple Example:

```bash
# request
GET /resources/public-data/ HTTP/1.1
Host: bar.other
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:71.0) Gecko/20100101 Firefox/71.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Language: en-us,en;q=0.5
Accept-Encoding: gzip,deflate
Connection: keep-alive
Origin: https://foo.example # Origin header

# response
HTTP/1.1 200 OK
Date: Mon, 01 Dec 2008 00:23:53 GMT
Server: Apache/2
Access-Control-Allow-Origin: * # requests from where are allowed
Keep-Alive: timeout=2, max=100
Connection: Keep-Alive
Transfer-Encoding: chunked
Content-Type: application/xml

[…XML Data…]
```

Preflight Example:

(When request contains non-standard headers & data)

```bash
## Preflight
# request
OPTIONS /doc HTTP/1.1
Host: bar.other
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:71.0) Gecko/20100101 Firefox/71.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Language: en-us,en;q=0.5
Accept-Encoding: gzip,deflate
Connection: keep-alive
Origin: http://foo.example
Access-Control-Request-Method: POST
Access-Control-Request-Headers: X-PINGOTHER, Content-Type
# response
HTTP/1.1 204 No Content
Date: Mon, 01 Dec 2008 01:15:39 GMT
Server: Apache/2
Access-Control-Allow-Origin: https://foo.example
Access-Control-Allow-Methods: POST, GET, OPTIONS
Access-Control-Allow-Headers: X-PINGOTHER, Content-Type
Access-Control-Max-Age: 86400
Vary: Accept-Encoding, Origin
Keep-Alive: timeout=2, max=100
Connection: Keep-Alive

## Real
# request
POST /doc HTTP/1.1
Host: bar.other
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:71.0) Gecko/20100101 Firefox/71.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Language: en-us,en;q=0.5
Accept-Encoding: gzip,deflate
Connection: keep-alive
X-PINGOTHER: pingpong
Content-Type: text/xml; charset=UTF-8
Referer: https://foo.example/examples/preflightInvocation.html
Content-Length: 55
Origin: https://foo.example
Pragma: no-cache
Cache-Control: no-cache

<person><name>Arun</name></person>
# response
HTTP/1.1 200 OK
Date: Mon, 01 Dec 2008 01:15:40 GMT
Server: Apache/2
Access-Control-Allow-Origin: https://foo.example
Vary: Accept-Encoding, Origin
Content-Encoding: gzip
Content-Length: 235
Keep-Alive: timeout=2, max=99
Connection: Keep-Alive
Content-Type: text/plain

[Some XML payload]
```

