# hashing algorithms

A function that maps **data of arbitrary size** to **a hash of a fixed size**.

* fast to compute for any size and kind of data.

* infeasible to invert (generate data from a given hash).

* infeasible to find two data with the same hash (No collision).

  note: there does exist collisions for any hash algorithm that outputs a fixed length hash (see **the pigeonhole principle**), but here we only require **it is probably impossible to find another data of the same hash with a given data/hash in limited time/future**.

* even smallest changes to the data should change the hash completely (Avalanche effect).

### MD5

the most usual algorithm for checking if two files are exactly the same.

We generate checksum/digest/signature of a file, and send the file with the checksum to the others. So the others can compute MD5 of the received file and compare it with the checksum to check if the received file is exactly the same as the sent file.

This is feasible because MD5 only has 128 bits, so it can be assumed to be invariant during the transmission.

MD5 is compromised and unsafe for cryptography! Now we can easily find collisions of MD5.

### SHA family

Secure Hash Algorithms. A large series of well known hash functions.

e.g., SHA-256


# asymmetric cryptography

also called public-key cryptography.

* public key: shared with others, to encrypt messages sent to me.
* private key: kept by myself, to decrypt messages encrypted by my public key.

asymmetric cryptosystem guarantees a secure message sending:

```
* A shares A's public key with B.
* B uses A's public key to encrypt the message.
* A uses A's private key to decrypt the message.
```

###  

Some widely used asymmetric cryptosystem:

* RSA
* DSA
* ECDSA (Elliptic Curve Digital Signature Algorithm)


### digital signature

asymmetric cryptosystem guarantees that **only you can read the messages sent to you**. （加密）

but there is still problems: 

* **how to verify the message is sent by a specific person?** （伪造）
* **how to verify the message is not modified after sent?** （篡改）

We can use bi-direction encryption & hash signature:

```
* A shares A's public key with B.
  B shares B's puclic key with A.
* B uses A's public key to encrypt the message.
  B uses B's private key to encrypt hash of the message [signature].
  B sends the encrypted message and the signature to A. 
* A uses A's private key to decrypt the message. (get message)
  A uses B's public key to decrypt the signature. (verify it is sent by B)
  A compares the hash of the received message with the signature. (verify the message is not modified after sent)
```

However, there is an even more severe problem:

* **How to verify the public key really belongs to a specific person?**

We have to include a 100% trusty third-person: **the certificate authority (CA)**.

It is a server like DNS server, to make clear each public key's owner by generating a Digital Certificate.

The DC is yet another encrypted message: it is **the specific person's public key encrypted by CA's private key**! So now we can use CA's public key to decrypt the DC and get the real public key of the specific person.

```
* A & B apply for DC of their public keys from CA.
* B uses A's DC to encrypt the message.
  B uses B's private key to encrypt the hash of the message [signature].
  B sends the encrypted message, the signature and B's DC to A.
* A uses A's private key to decrypt the message. (get message)
  A uses CA's public key to decrpyt B's DC. (get B's public key)
  A checks if B is the owner of this public key. (verify the public key belongs to B)
  A uses B's public key to decrypt the signature. (verify it is sent by B)
  A compares the hash of the received message with the signature. (verify the message is not modified after sent)
```

> HTTPS: an example.
>
> CA will record each website's public key. 
>
> (to enable https, the website should apply a pair of keys from CA.)
>
> ```
> * C requests a https connection to S.
> * S sends encrpyted message with the S's DC to C.
> * C uses CA's public key to decrypt S's DC. (get S's public key)
> * C checks if S is the owner of this public key from CA. (verify if this website is fake/pretended)
> * C uses S's public key to decrypt the message. (get message)
> ```