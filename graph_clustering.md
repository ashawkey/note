#### Graph clustering

there are different objective for a partition of a graph.

denotations:
$$
links(A, B) = \sum_{i \in A, j \in B}A_{ij} \\
degree(A) = links(A, V)\\
$$

* Ratio association
  $$
  RAssoc(G) = max_{V_1,...,V_k }\sum_{c=1}^k \frac {links(V_c, V_c)}{|V_c|} 
  $$

* Ratio cut
  $$
  RCut(G) = min_{V_1,...,V_k }\sum_{c=1}^k \frac {links(V_c, V \setminus V_c)}{|V_c|} \\
  $$
  if $V_1, ..., V_k$ are all of the same size, we call it the Kernighan-Lin objective.

* Normalized cut
  $$
  NCut(G) = min_{V_1,...,V_k }\sum_{c=1}^k \frac {links(V_c, V \setminus V_c)}{degree(V_c)} \\
  $$


* General Weighted graph cut/association

  denote $w(V_c) = \sum_{i \in V_c}w_i$ , use this to replace $|V_c|$ in $RAssoc$ and $RCut$.

  we use $WCut,  WAssoc$ to denote them.



#### Spectral clustering



#### Graclus multilevel clustering

Metis is another multilevel clustering algorithm.

* coarsening

  * heavy-edge coarsening

    metis uses this method, which works well for KL objective.

    ```python
    while not all vertex marked:
        pick an unmarked vertex v randomly
        find the heaviest edge started from v , to vertex w
        mark v and w as a supernode
        set supernode edge weight as v+w
    ```

  * max-cut coarsening

    generalization of heavy-edge with vertex weights

    Using $\frac {e(x,y)} {w(x)} +  \frac {e(x,y)} {w(y)}$ instead of simply $e(x,y)$.

  Stop when there are less than $5K$ nodes, $K$ is the targeted cluster number.

* base-clustering

  * region growing

    randomly select vertices and then BFS.

  * spectral clustering

  * bisection

    run coarsening until 20 nodes remained, then use kernel k-means.

* refinement

  rebuild $G_i$ for $G_{i+1} $ , then use kernel k-means to refine the boundary.
