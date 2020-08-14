# Security



### SSL: secure sockets layer

Protocols for establishing authenticated and encrypted links between networked computers.

The protocol is deprecated in 1999 with TLS release, but still used to refer to this technique.



### TLS: transport layer security

Successor of SSL.



### HTTPS

Tunnel HTTP over TSL/SSL which encrypts the HTTP payload.

**You need to buy certificates.**

... not cheap.

* For host:

  

* For visitor



### CSP: Content security policy

```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src https://*; child-src 'none';">
```

