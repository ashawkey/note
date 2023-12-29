# Report

### Introduction

* **Graph**: generic data representation form describing the geometric structures of data domains. 

  We denote an undirected, connected, weighted graph as $\mathcal{G} = \{\mathcal{V}, \mathcal{E}, \mathcal{W} \}$, and the number of vertices is $N = |\mathcal{V}|$.

* **Graph Signal**: data that reside on the vertices of a graph, usually represented as a vector $ f \in \mathbb{R}^N $. 

* **Edge Weights**: usually represent similarity between the two vertices.

  Common graph construction methods when edge weights are not naturally defined:

  * Gaussian kernel weighting function
  * k-nearest neighbors 

###  The Graph Laplacian

* **(Unnormalized) Graph Laplacian**:  $\mathcal{L} =D - W$

  $\mathcal{L}$ works as a difference operator for any signal $f$ . 
  

$$
\displaylines{

  Lf(i) = \sum_{j}w_{ij}(f(i)-f(j))
  
}
$$


  Since $\mathcal{L}$ is a real symmetric matrix for an undirected graph,  it has $N$ eigenvalues denoted as $\{\lambda_\iota\}_{\iota=0,1,...,N-1}​$. Also, since the graph has at least one connected component, there is at least one eigenvalue that is equal to zero.

* **Normalized Graph Laplacian**: 
  

$$
\displaylines{

  \hat{\mathcal{L}} = D^{-1/2}\mathcal{L}D^{-1/2} = I - D^{-1/2}WD^{-1/2}
  
}
$$


  the eigenvalues of $\hat{\mathcal{L}}$ belongs to $[0, 2]$ , with $\hat\lambda = 2$ if and only if $\mathcal{G}$ is bipartite.

*  **Random Walk Matrix**: $P = D^{-1}W$

There is not a clear answer as to when to use each of these basis.

### Graph Fourier Transform

The classical Fourier transform works in **the time domain and the frequency domain**. Analogously we define the graph Fourier transform, which works in **the vertex domain and the graph spectral domain**. 

For any signal $f$ in the vertex domain, we have its corresponding signal (kernel) $\hat{f}$ in the graph spectral domain.

In matrix form, we have $U$ ($n * n$) as the Fourier basis of the graph Laplacian (eigenvectors as columns), $F$  ($n * m$) as a list of $m$ signals on $n$ vertices (each signal is a column).

$$
\displaylines{
\hat{f}(\lambda_\iota) = \sum_{i=1}^{N}f(i)\mathcal{u}_\iota^*(i) \\
\hat{F} = U'F
}
$$

and the inverse graph Fourier transform:

$$
\displaylines{
f(i) = \sum_{\iota=0}^{N-1}\hat{f}(\lambda_\iota)u_\iota(i) \\
F = U\hat{F}
}
$$

The eigenvalues and eigenvectors in graph spectral domain provide a similar **notation of frequency**: the eigenvectors with larger eigenvalues contains more zero crossings.

### Discrete Calculus and Signal Smoothness

The edge derivative of a signal $f$ with respect to edge $e_{ij}$ at vertex $i$ is defined as :

$$
\displaylines{
\frac {\partial f} {\partial e} |_i = \sqrt{W_{i,j}}|f(j)-f(i)|
}
$$

The gradient of $f$ at vertex $i$  ($\nabla_if$) is therefore the vector of all derivatives with edges starting from $i$  in the graph.

The discrete $p$-Dirichlet form of $f$ is defined as:

$$
\displaylines{
S_p(f) =  \frac {1} {p} \sum_{i \in V}\parallel \nabla_if\parallel_2^p
}
$$

When $p=2$ , it is known as the **graph Laplacian quadratic form**.

Seminorm of $f$ is defined as $\parallel f\parallel _\mathcal{L} = \sqrt{S_2(f)}$ .

**The smoother the graph, the smaller the $S_2(f)$ .**

The discrete regularization framework:

$$
\displaylines{
{argmin}_f\{\parallel f-y\parallel_2^2 + \gamma S_p(f)\}
}
$$

When $p=2$ ,it is called Tikhonov Regularization and can be used for image denoising.

### Generalized Graph Signal Operators

##### Filtering

* graph spectral domain: 

  let $\hat{h}(1*n)$ be the transfer function in spectral domain. let $H = diag(\hat{h})$ .


$$
\displaylines{
\hat f_{out}(\lambda_\iota) = \hat f_{in}(\lambda_\iota)\hat h(\lambda_\iota) \\
\hat{F}_{out} = H  \hat{F}_{in}
}
$$


$ f_{out} $ can then be calculated through an inverse graph Fourier transform.

$$
\displaylines{
U' * F_{out} = H  U'  F_{in} \\
F_{out} = U  H  U'  F_{in}
}
$$


* vertex domain: linear combination within K-hop neighborhoods
  

$$
\displaylines{

  f_{out}(i) = b_{i,i}f_{in}(i) + \sum_{j \in \mathcal{N}(i,K)}b_{i,j}f_{in}(j)
  
}
$$


When the transfer function in graph spectral domain is an **order K polynomial**, the two forms of filtering can be related.

##### Convolution

**Convolution in the vertex domain (the time domain) is equivalent to multiplication in the graph spectral domain (the frequency domain).**

Since in the graph setting there is no $h(t-\tau)$, we use multiplication in the graph spectral domain to generalize the definition of convolution:

$$
\displaylines{
(f *_g h)(i) = \sum_{\iota = 0}^{N-1}\hat f(\lambda_\iota)\hat h(\lambda_\iota)u_\iota(i) \\
f *_g h = U((U'f) \odot (U'g)\\
F *_g h = UHU'F
}
$$


**so convolution with $h$ is just filter with $h$.**

##### Translation

It's not clear what it means to translate a graph signal, but we can generalize translation as a convolution with a delta centered at $n$:

$$
\displaylines{
(T_ng)(i) = \sqrt{N}(g*\delta_n)(i)
}
$$


##### Modulation 


$$
\displaylines{
(M_kg)(i) = \sqrt{N}u_k(i)g(i)
}
$$


##### Dilation


$$
\displaylines{
(\hat{\mathcal{D}_sg)}(\lambda) = \hat g(s\lambda)
}
$$


##### Coarsening

this operation can be separated in to two subtasks:

* Down Sampling: identifying a reduced set of vertices
* Reduction: assigning edges and weights

For a bipartite graph, it is natural to down sample it by a factor of 2.

For non-bipartite graphs, conditions are much more complex.

### Localized Multiscale Transforms

Wavelet transform can localize signal in both time and frequency simultaneously, which is better than Fourier transform.

* vertex domain
  * random transforms
  * graph wavelets
  * lifting-based transforms
  * tree wavelets
* graph spectral domain
  * diffusion wavelets
  * spectral graph wavelet
  * graph-quadrature mirror filterbanks

### Open Issues & Extension

##### Open Issues

* How the construction of graph affects properties of the localized, multiscale transforms for signals on graphs.
* When and why to use normalized, unnormalized graph Laplacian or other basis.
* What kind of distance to use in the vertex domain.
* Approximate computational techniques for signal processing on graphs.
* How to link structural properties of graph signals and their underlying graphs to properties.

##### Extensions

* Directed graph.
* Time series of signal data.
* Time series of underlying graphs.