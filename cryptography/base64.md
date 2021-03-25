# base64

A method that uses 64 printable characters (base64) to encode binary message to make it printable.

```
characters(url_safe): A-Z, a-z, 0-9, +(-), /(_)
padding character: =
```



It encodes every 3 byte binary string to 4 byte base64 string.

```python
chars = {i:c for i,c in enumerate(['A', ..., '/'])}
rev_chars = {v,k for k,v in chars.items()}

def base64_encode(b):
    # b is a binary string (by byte, so divisible by 8 bits)
    s = ''
	# pad 0 to make it divisible by 6 bits 
    # (some method also pads = to result s)
    if len(b) % 24 == 8:
        b += '0000'
    elif len(b) % 24 == 16:
        b += '00'
    # encode every 6 bits
    for i in range(0, len(b), 6):
        c = binary_to_int(b[i:i+6])
        s += chars[c]
    return s

def base64_decode(s):
    b = ''
    for c in s:
        b += int_to_binary(c)
    return b  
```





### python API

```python
import base64

bs = b'abinarystring'
bs64 = base64.b64encode(bs) # b'YWJpbmFyeXN0cmluZw=='

bs == base64.b64decode(bs64) # True
```



Note the difference between bytes and strings.

```python
# Bytes vs String

s = 'test'
bs = b'test'

str.encode(s) == s.encode() == bs # True
bs.decode() == s # True

```

