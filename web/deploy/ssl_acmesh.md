# SSL



### SSL: secure sockets layer

Protocols for establishing authenticated and encrypted links between networked computers.

The protocol is deprecated in 1999 with TLS release, but still used to refer to this technique.



### TLS: transport layer security

Successor of SSL.



### HTTPS

Tunnel HTTP over TSL/SSL which encrypts the HTTP payload.

**You need to buy certificates.**

Or use free license by `acme.sh`  !



### acme.sh

https://github.com/acmesh-official/acme.sh/wiki/%E8%AF%B4%E6%98%8E

```bash
curl  https://get.acme.sh | sh
alias acme.sh=~/.acme.sh/acme.sh

# setup http for nginx (i.e, the website can be accessed by http), so acme.sh can verify your identity.
# server {listen 80; ...} 

# issue a license for nginx
# -d must be in nginx config (`server_name www.kiui.moe;`)
# Some times it reports `can't get nonce` or looping `sleep 10 and retry`, just re-run the code several times :)
acme.sh --issue -d www.kiui.moe --nginx # --debug 2

# install cert
acme.sh --install-cert \
        -d www.kiui.com \
        --key-file /etc/nginx/key.pem  \
        --fullchain-file /etc/nginx/cert.pem \
        --reloadcmd "systemctl restart nginx"

# setup https for nginx. (in manual)
# server {listen 443 ssl; ...}
```





### CSP: Content security policy

```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src https://*; child-src 'none';">
```

