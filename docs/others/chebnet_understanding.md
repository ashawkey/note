#### Convolution

convolution in spectral domain is just a frequency filter.

let the graph Laplacian is $n * n$.
$$
y = UHU'x = Ug_\theta(\lambda)U'x = g_\theta(L)x
$$
but to learn a filter's complicity is $O(n)$. (there are $n$ free parameters)

the problem is how to make it faster.

we need parametrization of the filter.

we use Chebyshev polynomial to approximate it.
$$
g_\theta(\lambda) = diag(\theta) \rightarrow \sum_{k=0}^{K-1}\theta_k\Lambda^k \rightarrow  \sum_{k=0}^{K-1}\theta_kT_k(\tilde\Lambda) \\
\Lambda = diag(\lambda)\\
g_\theta(L) \rightarrow Ug_\theta(\lambda)U' =  \sum_{k=0}^{K-1}\theta_kT_k(\tilde L) \\
then \ \ \ y = g_\theta(L)x \\
let \ \ \bar{x_k} = T_k(\tilde L)x \\
y = \bar{X}\theta \\
$$
due to the property of Chebyshev:
$$
\bar{x_k} = 2\tilde L \bar{x_{k-1}} - \bar{x_{k-2}} \\
\tilde L = 2L/\lambda_{max} - I
$$
so we avoided computing eigenvalues and Fourier basis $U$ , and just use $L$ and $\lambda_{max}$ to recursively calculate $y$ from $x$.

time complexity is from $O(n^2)$ to $O(K|\mathcal E|)$



#### Pooling

pooling is clustering or cutting for a graph.

graph clustering is NP-hard.

we need multi-level clustering: Graclus, which also need not calculating eigenvalues.

* Graclus
  * coarsening
  * base-clustering: kernel k-means or spectral clustering
  * refinement: kernel k-means

Then we pool it by adding fake nodes to build a balanced binary tree.

Finally, we feed it to a Fully connected Layer for Output.



#### Codes

* graph

  ```python
  grid(m, dtype=np.float32)
  '''
  return a meshgrid of [m*m, 2] in [0,1] (uniformly divided by m)
  '''
  
  distance_*(z, k=4, metirc = "euclidean")
  '''
  input
  	z is a N*M matrix of N examples in dimension M.
  	k is the kNN parameter.
  output
  	d: N*k matrix, each row is the closest k distance.
  	idx: N*k matrix, each row is the closest k index.
  '''
  
  adjacency(dist, idx)
  '''
  input
  	output of dist_*()
  output
  	adjacency matrix of N*N.
  	in fact it is a **Mutral** K-NN graph.
  '''
  
  replace_random_edges(A, noise_level)
  '''
  just add random noise.
  '''
  
  laplacian(W, normalized=True)
  '''
  input
  	Adjacency matrix N*N
  output
  	graph laplacian
  '''
  
  lmax(L, normalized=True)
  '''
  compute Lmax for Chebyshev approximation.
  if not normalized, only calc the largest Eigenvalue.
  '''
  
  fourier(L, algo='eigh', k=1)
  '''
  return sorted lamb,U
  in fact, not used in standard cnn_graph.
  '''
  
  plot_spectrum(L, algo="eig")
  '''
  input: L is a list of Laplacians.
  output
  	plot of lamb values.
  '''
  
  rescale_L(L, lmax=2)
  '''
  scale L into [-1, 1], for chebyshev approx.
  \tilde L = 2*L/lmax - I
  '''
  
  chebyshev(L, X, K)
  '''
  input
  	L: N*N graph laplacian, need to be normalized and rescaled ?
  	X: N*M matrix
  	K: int, max order
  output
  	Xt: K*N*M matrix. Xt[k] is the k order chebyshev 
  '''
  
  lanczos(L, X, K)
  '''
  
  '''
  
  ```

* coarsening

  ```python
  metis(W, levels, rid=None)
  '''
  input
  	W: N*N adjacency matrix.
  	levels: to coarsen.
  	rid: a permutation list of N, if None, set to random.
  output
  	graphs: #levels*1 vector of multi-level Adjacency matrix. graphs[0] is W. 
  	Supernode' weight is sum of links between its child nodes.
  	parents: #(levels-1)*1 vector, cluster_id for each graph.
  '''
  
  metis_one_level(rr,cc,vv,rid,weights)
  '''
  input
  	rr,cc,vv: ordered non-zero entries of W(N*N).
  	rid: as in metis.
  	weights: N*1 vector, degree of each node.
  output
  	cluster_id: N*1 vector. each node's label of supernode.
  '''
  
  compute_perm(parents)
  '''
  input: output of metis.
  output: reordered permutation to satisfy a binary tree.
  	by finding parents and adding fake nodes, we build the unique tree from a $parents list.
  	
  assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
          == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])
  '''
  
  perm_adjacency(A, indices)
  '''
  input
  	A: adjacency matrix
  	indices: eg.compute_perms(parents)[0]
  output
  	A: adjacency after adding fake nodes, and sorted by indices.
  '''
  
  coarsen(A, levels, self_connections=False)
  '''
  input
  	A, levels : as in metis
  	self_connections: whether to use self_connection(diag of Adjacency), unimplemented ?
  output
  	graphs: as in metis, but sorted and fake nodes added.
  	perms[0]: nodes' order of graph[0]
  '''
  
  
  perm_data(X, indices)
  '''
  input
  	X: dataset, N*M
  	indices: perm[0], returned by coarsen()
  output
  	Xnew: dataset sorted and fake nodes added.
  '''
  ```

* models

  ```python
  # tensorflow models
  class base_model:
      predict(data)
      evaluate(data, labels)
      fit(train_Data, train_labels, val_data, val_labels)
      get_var(name)
      build_graph(M_0)
      inference(data, dropout)
      probabilities(logits)
      predition(logits)
      loss(logits, labels, regularization)
      training(loss, lr, ds, ...)
      
      
  class cgcnn(base_model):
      # chebyshev approx
      __init__(self, **params)
      '''
      L: normalized graph Laplacian
      F: number of graph conv filters, eg.[32, 64]
      K: polynomial orders, eg.[20, 20]
      p: pooling size eg.[4, 2]
      M: output dim of FC layers, eg.[512, y.max()+1]
      
      dir_name='' : output directory
      
      filter='chebyshev5' :
      brelu='b1relu' :
      pool='mpool1' :
      num_epochs=20 :
      learning_rate=0.1 :
      decay_rate=0.95 :
      decay_steps=None :  eg. n_train/batch_size
      momentum=0.9 :
      regularization=0 :
      dropout=0 :
      batch_size=100 :
      eval_frequenct=200 : eg.30*epoch
      
      ###
      the structure of the NN is defined as:
      for i in F,P,K
      	* conv of F[i] out_channels
      		the chebyshev poly approx order is K[i]
      	* pool of P[i] pool_size
      for i in M
      	* fc of M[i] out_dims
     		(usually the last of M is n_Class)
      
      '''
  
      
  # for some examples of parameters:
  common = {}
  common['dir_name']       = 'mnist/'
  common['num_epochs']     = 20
  common['batch_size']     = 100
  common['decay_steps']    = mnist.train.num_examples / common['batch_size']
  common['eval_frequency'] = 30 * common['num_epochs']
  common['brelu']          = 'b1relu'
  common['pool']           = 'mpool1'
  C = max(mnist.train.labels) + 1  # number of classes
  model_perf = utils.model_perf()
  
  if True:
      name = 'softmax'
      params = common.copy()
      params['dir_name'] += name
      params['regularization'] = 5e-4
      params['dropout']        = 1
      params['learning_rate']  = 0.02
      params['decay_rate']     = 0.95
      params['momentum']       = 0.9
      params['F']              = []
      params['K']              = []
      params['p']              = []
      params['M']              = [C]
      model_perf.test(models.cgcnn(L, **params), name, params,
                      train_data, train_labels, val_data, val_labels, test_data, test_labels)
      '''
      NN architecture
  	  input: M_0 = 1040
  	  layer 1: logits (softmax)
  	    representation: M_1 = 10
      	weights: M_0 * M_1 = 1040 * 10 = 10400
     		biases: M_1 = 10
      '''
      
  # Common hyper-parameters for networks with one convolutional layer.
  common['regularization'] = 0
  common['dropout']        = 1
  common['learning_rate']  = 0.02
  common['decay_rate']     = 0.95
  common['momentum']       = 0.9
  common['F']              = [10]
  common['K']              = [20]
  common['p']              = [1]
  common['M']              = [C]
  if True:
      name = 'cgconv_softmax'
      params = common.copy()
      params['dir_name'] += name
      params['filter'] = 'chebyshev5'
      model_perf.test(models.cgcnn(L, **params), name, params,
                      train_data, train_labels, val_data, val_labels, test_data, test_labels)
      
      '''
      NN architecture
  	  input: M_0 = 1040
   	  layer 1: cgconv1
      	representation: M_0 * F_1 / p_1 = 1040 * 10 / 1 = 10400
   	   weights: F_0 * F_1 * K_1 = 1 * 10 * 20 = 200
   	   biases: F_1 = 10
  	  layer 2: logits (softmax)
   	   representation: M_2 = 10
   	   weights: M_1 * M_2 = 10400 * 10 = 104000
  	    biases: M_2 = 10
      '''
      
  # Common hyper-parameters for LeNet5-like networks.
  common['regularization'] = 5e-4
  common['dropout']        = 0.5
  common['learning_rate']  = 0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
  common['decay_rate']     = 0.95
  common['momentum']       = 0.9
  common['F']              = [32, 64]
  common['K']              = [25, 25]
  common['p']              = [4, 4]
  common['M']              = [512, C]
  if True:
      name = 'cgconv_cgconv_fc_softmax'  # 'Chebyshev'
      params = common.copy()
      params['dir_name'] += name
      params['filter'] = 'chebyshev5'
      model_perf.test(models.cgcnn(L, **params), name, params,
                      train_data, train_labels, val_data, val_labels, test_data, test_labels)
  '''
  NN architecture
    input: M_0 = 1008
    layer 1: cgconv1
      representation: M_0 * F_1 / p_1 = 1008 * 32 / 4 = 8064
      weights: F_0 * F_1 * K_1 = 1 * 32 * 25 = 800
      biases: F_1 = 32
    layer 2: cgconv2
      representation: M_1 * F_2 / p_2 = 252 * 64 / 4 = 4032
      weights: F_1 * F_2 * K_2 = 32 * 64 * 25 = 51200
      biases: F_2 = 64
    layer 3: fc1
      representation: M_3 = 512
      weights: M_2 * M_3 = 4032 * 512 = 2064384
      biases: M_3 = 512
    layer 4: logits (softmax)
      representation: M_4 = 10
      weights: M_3 * M_4 = 512 * 10 = 5120
      biases: M_4 = 10
  '''
  ```

* utils

```python
grid_search(params, grid_params, train_data, train_labels, val_data, val_labels, test_data, test_labels, model)
'''

'''

class model_perf:
    test()
    show()
    '''
    used for testing many structures and show their performance together.
    '''
```



#### Example









