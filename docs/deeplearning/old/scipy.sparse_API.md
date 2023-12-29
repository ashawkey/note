#### Sparse matrix class

* bsr_matrix

  Block Sparse Row format

* coo_matrix

  Coordinate format. 

  No support for arithmetic operators, and no slicing.

  **Usually used to construct, then changed to csr/csc for arithmetic.**

  ```python
  coo_matrix(D)
  coo_matrix((m, n), dtype=np.int32)
  coo_matrix((data, (I, J)), shape=[M, N])
  
  coo.toarray()
  coo.shape/ndim/dtype
  coo.tocsr() # csr_matrix(coo)
  ```

* csr_matrix

  Compressed Sparse Row format

  **efficient row slicing and arithmetic operations between CSRs.**

* csc_matrix

  Compressed Sparse Column format

* dia_matrix

  Diagonal storage format

  ```python
  dia_matrix((data, offsets=0), shape=(M,N))
  ```

* dok_matrix

  Dictionary Of Key format

* lil_matrix

  Linked List format

#### Functions

```python
eye(m[, dtype, format="dia"])
identity(n[, dtype, format])
diags(diagonals[, offsets, shape, format, dtype])
rand(m, n[, density, format, dtype])
issparse(x) # isspmatrix(x)
isspmatrix_csc(x)
```

#### linalg

```python
inv(A)
expm(A) # exp using Pade approximation

eigs(A[, k, M, which])
	'''
	k: calculate the first k eig
	M: generalized eigenproblem Ax=wMx
	which: which first k eig to find
		"LM": largest magnitude
		"SM": smallest magnitude
		"LR": largest real part
		"LI": largest imaginary part
	'''
eigsh(A[, k]) # fast for real sym mat

```


