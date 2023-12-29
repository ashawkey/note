# Image Processing Basics

### HSV

Hue(360), Saturation(1), Value(1)

```python
 BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv


def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)

	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out
```


### Histogram

```python
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
```


### Gamma Correction


$$
\displaylines{
I_{out} ={\frac{1}{c}\ I_{in}} ^ {\frac{1}{g}}
}
$$


校正照相机等电子设备传感器的非线性光电转换特征，主要是增大RGB值。

```python
def gamma_correction(img, c=1, g=2.2):
	out = img.copy()
	out /= 255.
	out = (1/c * out) ** (1/g)

	out *= 255
	out = out.astype(np.uint8)

	return out
```


### Interpolation

* Nearest Neighbor

  ```python
  def nn_interpolate(img, ax=1, ay=1):
      H, W, C = img.shape
      
      aH = int(H*ay)
      aW = int(W*ax)
      
  	y = np.arange(aH).repeat(aW).reshape(aW, -1)
      x = np.tile(np.arange(aW), (aH, 1))
      
      y = np.round(y / ay).astype(np.int32)
      x = np.round(x / ax).astype(np.int32)
      
      out = img[y, x].astype(np.uint8)
      
      return out
      
  ```

* Bilinear

  ```python
  def bl_interpolate(img, ax=1., ay=1.):
  	H, W, C = img.shape
  
  	aH = int(ay * H)
  	aW = int(ax * W)
  
  	# get position of resized image
  	y = np.arange(aH).repeat(aW).reshape(aW, -1)
  	x = np.tile(np.arange(aW), (aH, 1))
  
  	# get position of original position
  	y = (y / ay)
  	x = (x / ax)
  
  	ix = np.floor(x).astype(np.int)
  	iy = np.floor(y).astype(np.int)
  
  	ix = np.minimum(ix, W-2)
  	iy = np.minimum(iy, H-2)
  
  	# get distance 
  	dx = x - ix
  	dy = y - iy
  
  	dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
  	dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
  
  	# interpolation
  	out = (1-dx) * (1-dy) * img[iy, ix] + 
      	  dx * (1 - dy) * img[iy, ix+1] + 
            (1 - dx) * dy * img[iy+1, ix] + 
            dx * dy * img[iy+1, ix+1]
  
  	out = np.clip(out, 0, 255)
  	out = out.astype(np.uint8)
  
  	return out
  ```

* Bicubic

  ```python
  def bc_interpolate(img, ax=1., ay=1.):
  	H, W, C = img.shape
  
  	aH = int(ay * H)
  	aW = int(ax * W)
  
  	# get positions of resized image
  	y = np.arange(aH).repeat(aW).reshape(aW, -1)
  	x = np.tile(np.arange(aW), (aH, 1))
  	y = (y / ay)
  	x = (x / ax)
  
  	# get positions of original image
  	ix = np.floor(x).astype(np.int)
  	iy = np.floor(y).astype(np.int)
  
  	ix = np.minimum(ix, W-1)
  	iy = np.minimum(iy, H-1)
  
  	# get distance of each position of original image
  	dx2 = x - ix
  	dy2 = y - iy
  	dx1 = dx2 + 1
  	dy1 = dy2 + 1
  	dx3 = 1 - dx2
  	dy3 = 1 - dy2
  	dx4 = 1 + dx3
  	dy4 = 1 + dy3
  
  	dxs = [dx1, dx2, dx3, dx4]
  	dys = [dy1, dy2, dy3, dy4]
  
  	# bi-cubic weight
  	def weight(t):
  		a = -1.
  		at = np.abs(t)
  		w = np.zeros_like(t)
  		ind = np.where(at <= 1)
  		w[ind] = ((a+2) * np.power(at, 3) - (a+3) * np.power(at, 2) + 1)[ind]
  		ind = np.where((at > 1) & (at <= 2))
  		w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
  		return w
  
  	w_sum = np.zeros((aH, aW, C), dtype=np.float32)
  	out = np.zeros((aH, aW, C), dtype=np.float32)
  
  	# interpolate
  	for j in range(-1, 3):
  		for i in range(-1, 3):
  			ind_x = np.minimum(np.maximum(ix + i, 0), W-1)
  			ind_y = np.minimum(np.maximum(iy + j, 0), H-1)
  
  			wx = weight(dxs[i+1])
  			wy = weight(dys[j+1])
  			wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
  			wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)
  
  			w_sum += wx * wy
  			out += wx * wy * img[ind_y, ind_x]
  
  	out /= w_sum
  	out = np.clip(out, 0, 255)
  	out = out.astype(np.uint8)
  
  	return out
  ```


### Affine Transform


$$
\displaylines{
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
a&b&t_x\\
c&d&t_y\\
0&0&1
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
\\
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)=
\frac{1}{a\  d-b\  c}\ 
\left(
\begin{matrix}
d&-b\\
-c&a
\end{matrix}
\right)\  
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)-
\left(
\begin{matrix}
t_x\\
t_y
\end{matrix}
\right)
}
$$


```python
def affine(img, a, b, c, d, tx, ty):
  	H, W, C = img.shape

	# temporary image
	img = np.zeros((H+2, W+2, C), dtype=np.float32)
	img[1:H+1, 1:W+1] = _img

	# get new image shape
	H_new = np.round(H * d).astype(np.int)
	W_new = np.round(W * a).astype(np.int)
	out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image
	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# get position of original image by reverse-affine
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

	x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
	y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixel to new image
	out[y_new, x_new] = img[y, x]

	out = out[:H_new, :W_new]
	out = out.astype(np.uint8)

	return out
```

* Shift
  

$$
\displaylines{

  \left(
  \begin{matrix}
  x'\\
  y'\\
  1
  \end{matrix}
  \right)=
  \left(
  \begin{matrix}
  1&0&t_x\\
  0&1&t_y\\
  0&0&1
  \end{matrix}
  \right)\  
  \left(
  \begin{matrix}
  x\\
  y\\
  1
  \end{matrix}
  \right)
  
}
$$


* Resize
  

$$
\displaylines{

  \left(
  \begin{matrix}
  x'\\
  y'\\
  1
  \end{matrix}
  \right)=
  \left(
  \begin{matrix}
  ax&0&t_x\\
  0&ay&t_y\\
  0&0&1
  \end{matrix}
  \right)\  
  \left(
  \begin{matrix}
  x\\
  y\\
  1
  \end{matrix}
  \right)
  
}
$$


  
* Rotate
  

$$
\displaylines{

  \left(
  \begin{matrix}
  x'\\
  y'\\
  1
  \end{matrix}
  \right)=
  \left(
  \begin{matrix}
  \cos(A)&-\sin(A)&t_x\\
  \sin(A)&\cos(A)&t_y\\
  0&0&1
  \end{matrix}
  \right)\ 
  \left(
  \begin{matrix}
  x\\
  y\\
  1
  \end{matrix}
  \right)
  
}
$$


  
* Sharing
  

$$
\displaylines{

  a=\frac{t_x}{h}\\
    \left[
    \begin{matrix}
    x'\\
    y'\\
    1
    \end{matrix}
    \right]=\left[
    \begin{matrix}
    1&a&t_x\\
    0&1&t_y\\
    0&0&1
    \end{matrix}
    \right]\ 
    \left[
    \begin{matrix}
    x\\
    y\\
    1
    \end{matrix}
    \right]
  \\
    a=\frac{t_y}{w}\\
    \left[
    \begin{matrix}
    x'\\
    y'\\
    1
    \end{matrix}
    \right]=\left[
    \begin{matrix}
    1&0&t_x\\
    a&1&t_y\\
    0&0&1
    \end{matrix}
    \right]\ 
    \left[
    \begin{matrix}
    x\\
    y\\
    1
    \end{matrix}
    \right]
  
}
$$


### Fourier Transform


$$
\displaylines{
G(k,l)=\frac{1}{H\  W}\ \sum\limits_{y=0}^{H-1}\ \sum\limits_{x=0}^{W-1}\ I(x,y)\ e^{-2\  \pi\  j\ (\frac{k\  x}{W}+\frac{l\  y}{H})} \\
I(x,y)=\frac{1}{H\  W}\ \sum\limits_{l=0}^{H-1}\ \sum\limits_{k=0}^{W-1}\ G(l,k)\ e^{2\  \pi\  j\ (\frac{k\  x}{W}+\frac{l\  y}{H})}
}
$$


```python
# DFT hyper-parameters
K, L = 128, 128
channel = 3

# DFT
def dft(img):
	H, W, _ = img.shape

	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out
```

在图像中，高频成分指的是颜色改变的地方（噪声或者轮廓等），低频成分指的是颜色不怎么改变的部
分（比如落日的渐变）。

* Low-pass

  ```python
  # LPF
  def lpf(G, ratio=0.5):
  	H, W, _ = G.shape	
  
  	# transfer positions
  	_G = np.zeros_like(G)
  	_G[:H//2, :W//2] = G[H//2:, W//2:]
  	_G[:H//2, W//2:] = G[H//2:, :W//2]
  	_G[H//2:, :W//2] = G[:H//2, W//2:]
  	_G[H//2:, W//2:] = G[:H//2, :W//2]
  
  	# get distance from center (H / 2, W / 2)
  	x = np.tile(np.arange(W), (H, 1))
  	y = np.arange(H).repeat(W).reshape(H, -1)
  
  	# make filter
  	_x = x - W // 2
  	_y = y - H // 2
  	r = np.sqrt(_x ** 2 + _y ** 2)
  	mask = np.ones((H, W), dtype=np.float32)
  	mask[r > (W // 2 * ratio)] = 0
  
  	mask = np.repeat(mask, channel).reshape(H, W, channel)
  
  	# filtering
  	_G *= mask
  
  	# reverse original positions
  	G[:H//2, :W//2] = _G[H//2:, W//2:]
  	G[:H//2, W//2:] = _G[H//2:, :W//2]
  	G[H//2:, :W//2] = _G[:H//2, W//2:]
  	G[H//2:, W//2:] = _G[:H//2, :W//2]
  
  	return G
  ```

* High-pass

  ```python
  # HPF
  def hpf(G, ratio=0.1):
  	H, W, _ = G.shape	
  
  	# transfer positions
  	_G = np.zeros_like(G)
  	_G[:H//2, :W//2] = G[H//2:, W//2:]
  	_G[:H//2, W//2:] = G[H//2:, :W//2]
  	_G[H//2:, :W//2] = G[:H//2, W//2:]
  	_G[H//2:, W//2:] = G[:H//2, :W//2]
  
  	# get distance from center (H / 2, W / 2)
  	x = np.tile(np.arange(W), (H, 1))
  	y = np.arange(H).repeat(W).reshape(H, -1)
  
  	# make filter
  	_x = x - W // 2
  	_y = y - H // 2
  	r = np.sqrt(_x ** 2 + _y ** 2)
  	mask = np.ones((H, W), dtype=np.float32)
  	mask[r < (W // 2 * ratio)] = 0
  
  	mask = np.repeat(mask, channel).reshape(H, W, channel)
  
  	# filtering
  	_G *= mask
  
  	# reverse original positions
  	G[:H//2, :W//2] = _G[H//2:, W//2:]
  	G[:H//2, W//2:] = _G[H//2:, :W//2]
  	G[H//2:, :W//2] = _G[:H//2, W//2:]
  	G[H//2:, W//2:] = _G[:H//2, :W//2]
  
  	return G
  ```

* Band-pass

  ```python
  # BPF
  def bpf(G, ratio1=0.1, ratio2=0.5):
  	H, W, _ = G.shape	
  
  	# transfer positions
  	_G = np.zeros_like(G)
  	_G[:H//2, :W//2] = G[H//2:, W//2:]
  	_G[:H//2, W//2:] = G[H//2:, :W//2]
  	_G[H//2:, :W//2] = G[:H//2, W//2:]
  	_G[H//2:, W//2:] = G[:H//2, :W//2]
  
  	# get distance from center (H / 2, W / 2)
  	x = np.tile(np.arange(W), (H, 1))
  	y = np.arange(H).repeat(W).reshape(H, -1)
  
  	# make filter
  	_x = x - W // 2
  	_y = y - H // 2
  	r = np.sqrt(_x ** 2 + _y ** 2)
  	mask = np.ones((H, W), dtype=np.float32)
  	mask[(r < (W // 2 * ratio1)) | (r > (W // 2 * ratio2))] = 0
  
  	mask = np.repeat(mask, channel).reshape(H, W, channel)
  
  	# filtering
  	_G *= mask
  
  	# reverse original positions
  	G[:H//2, :W//2] = _G[H//2:, W//2:]
  	G[:H//2, W//2:] = _G[H//2:, :W//2]
  	G[H//2:, :W//2] = _G[:H//2, W//2:]
  	G[H//2:, W//2:] = _G[:H//2, :W//2]
  
  	return G
  ```

  

### JPEG Compression

1. 将图像从RGB色彩空间变换到YCbCr色彩空间；
2. 对YCbCr做DCT；
3. DCT之后做量化；
4. 量化之后应用IDCT；
5. IDCT之后从YCbCr色彩空间变换到RGB色彩空间。


```python
T = 8
K = 8
channel = 3


# BGR -> Y Cb Cr
def BGR2YCbCr(img):
  H, W, _ = img.shape

  ycbcr = np.zeros([H, W, 3], dtype=np.float32)

  ycbcr[..., 0] = 0.2990 * img[..., 2] + 0.5870 * img[..., 1] + 0.1140 * img[..., 0]
  ycbcr[..., 1] = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
  ycbcr[..., 2] = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

  return ycbcr

# Y Cb Cr -> BGR
def YCbCr2BGR(ycbcr):
  H, W, _ = ycbcr.shape

  out = np.zeros([H, W, channel], dtype=np.float32)
  out[..., 2] = ycbcr[..., 0] + (ycbcr[..., 2] - 128.) * 1.4020
  out[..., 1] = ycbcr[..., 0] - (ycbcr[..., 1] - 128.) * 0.3441 - (ycbcr[..., 2] - 128.) * 0.7139
  out[..., 0] = ycbcr[..., 0] + (ycbcr[..., 1] - 128.) * 1.7718

  out = np.clip(out, 0, 255)
  out = out.astype(np.uint8)

  return out


# DCT weight
def DCT_w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * DCT_w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * DCT_w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out

# Quantization
def quantization(F):
    H, W, _ = F.shape

    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                (12, 12, 14, 19, 26, 58, 60, 55),
                (14, 13, 16, 24, 40, 57, 69, 56),
                (14, 17, 22, 29, 51, 87, 80, 62),
                (18, 22, 37, 56, 68, 109, 103, 77),
                (24, 35, 55, 64, 81, 104, 113, 92),
                (49, 64, 78, 87, 103, 121, 120, 101),
                (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    for ys in range(0, H, T):
        for xs in range(0, W, T):
            for c in range(channel):
                F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q

    return F


# JPEG without Hufman coding
def JPEG(img):
    ycbcr = BGR2YCbCr(img) # BGR -> Y Cb Cr
    F = dct(ycbcr) # DCT
    F = quantization(F) # quantization
    ycbcr = idct(F) # IDCT
    out = YCbCr2BGR(ycbcr) # Y Cb Cr -> BGR

    return out
```


### Canny Edge Detector

```python
def Canny(img):

	# Gray scale
	def BGR2GRAY(img):
		b = img[:, :, 0].copy()
		g = img[:, :, 1].copy()
		r = img[:, :, 2].copy()

		# Gray scale
		out = 0.2126 * r + 0.7152 * g + 0.0722 * b
		out = out.astype(np.uint8)

		return out


	# Gaussian filter for grayscale
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
		#K /= (sigma * np.sqrt(2 * np.pi))
		K /= (2 * np.pi * sigma * sigma)
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

		out = np.clip(out, 0, 255)
		out = out[pad : pad + H, pad : pad + W]
		out = out.astype(np.uint8)

		if gray:
			out = out[..., 0]

		return out


	# sobel filter
	def sobel_filter(img, K_size=3):
		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad : pad + H, pad : pad + W]
		out_v = out_v.astype(np.uint8)
		out_h = out_h[pad : pad + H, pad : pad + W]
		out_h = out_h.astype(np.uint8)

		return out_v, out_h


	def get_edge_angle(fx, fy):
		# get edge strength
		edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
		edge = np.clip(edge, 0, 255)

		fx = np.maximum(fx, 1e-10)
		#fx[np.abs(fx) <= 1e-5] = 1e-5

		# get edge angle
		angle = np.arctan(fy / fx)

		return edge, angle


	def angle_quantization(angle):
		angle = angle / np.pi * 180
		angle[angle < -22.5] = 180 + angle[angle < -22.5]
		_angle = np.zeros_like(angle, dtype=np.uint8)
		_angle[np.where(angle <= 22.5)] = 0
		_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
		_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
		_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

		return _angle


	def non_maximum_suppression(angle, edge):
		H, W = angle.shape
		_edge = edge.copy()
		
		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
							_edge[y, x] = 0

		return _edge

	def hysterisis(edge, HT=100, LT=30):
		H, W = edge.shape

		# Histeresis threshold
		edge[edge >= HT] = 255
		edge[edge <= LT] = 0

		_edge = np.zeros((H + 2, W + 2), dtype=np.float32)
		_edge[1 : H + 1, 1 : W + 1] = edge

		## 8 - Nearest neighbor
		nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

		for y in range(1, H+2):
				for x in range(1, W+2):
						if _edge[y, x] < LT or _edge[y, x] > HT:
								continue
						if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
								_edge[y, x] = 255
						else:
								_edge[y, x] = 0

		edge = _edge[1:H+1, 1:W+1]
								
		return edge

	# grayscale
	gray = BGR2GRAY(img)
	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)
	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)
	# get edge strength, angle
	edge, angle = get_edge_angle(fx, fy)
	# angle quantization
	angle = angle_quantization(angle)
	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)
	# hysterisis threshold
	out = hysterisis(edge, 50, 20)

	return out
```


### Hough Transform (Line detection)

```python
def Hough_Line(edge, img):
	## Voting
	def voting(edge):
		H, W = edge.shape
		
		drho = 1
		dtheta = 1

		# get rho max length
		rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

		# hough table
		hough = np.zeros((rho_max * 2, 180), dtype=np.int)

		# get index of edge
		ind = np.where(edge == 255)

		## hough transformation
		for y, x in zip(ind[0], ind[1]):
				for theta in range(0, 180, dtheta):
						# get polar coordinat4s
						t = np.pi / 180 * theta
						rho = int(x * np.cos(t) + y * np.sin(t))

						# vote
						hough[rho + rho_max, theta] += 1
							
		out = hough.astype(np.uint8)

		return out

	# non maximum suppression
	def non_maximum_suppression(hough):
		rho_max, _ = hough.shape

		## non maximum suppression
		for y in range(rho_max):
			for x in range(180):
				# get 8 nearest neighbor
				x1 = max(x-1, 0)
				x2 = min(x+2, 180)
				y1 = max(y-1, 0)
				y2 = min(y+2, rho_max-1)
				if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
					pass
					#hough[y,x] = 255
				else:
					hough[y,x] = 0

		return hough

	def inverse_hough(hough, img):
		H, W, _ = img.shape
		rho_max, _ = hough.shape

		out = img.copy()

		# get x, y index of hough table
		ind_x = np.argsort(hough.ravel())[::-1][:20]
		ind_y = ind_x.copy()
		thetas = ind_x % 180
		rhos = ind_y // 180 - rho_max / 2

		# each theta and rho
		for theta, rho in zip(thetas, rhos):
			# theta[radian] -> angle[degree]
			t = np.pi / 180. * theta

			# hough -> (x,y)
			for x in range(W):
				if np.sin(t) != 0:
					y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
					y = int(y)
					if y >= H or y < 0:
						continue
					out[y, x] = [0, 0, 255]
			for y in range(H):
				if np.cos(t) != 0:
					x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
					x = int(x)
					if x >= W or x < 0:
						continue
					out[y, x] = [0, 0, 255]
				
		out = out.astype(np.uint8)

		return out


	# voting
	hough = voting(edge)
	# non maximum suppression
	hough = non_maximum_suppression(hough)
	# inverse hough
	out = inverse_hough(hough, img)

	return out
```


### Dilate

```python
# Morphology Dilate
def Morphology_Dilate(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
					out[y-1, x-1] = 0

	return out

```

### Erode

```python
# Morphology Erode
def Morphology_Erode(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
					out[y-1, x-1] = 255

	return out
```


### Opening Operation

Dilate N times, Erode N times. ==> Remove isolated pixels.

### Closing Operation

Erode N times, Dilate N times. ==> Connect discrete pixels.

