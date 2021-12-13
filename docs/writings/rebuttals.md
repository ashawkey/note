Thanks for the reviewers’ constructive comments. As recognized by the reviewers, we address the problem of graph learning from incomplete graph data based on similarity preservation in the feature space. The effectiveness of our method is validated on various datasets, with low latency and memory consumption.

===
[Common Question] of <Reviewer 2 & 3>: More discussion and interpretation to the results and why our method performs better.

We discuss the experimental results from the following three aspects.
1) Compared with GCN and GMNN with fixed graphs as input, we outperform by 3.5% and 0.9% on average respectively on three datasets. This is because fixed graphs capture limited similarities between each pair of nodes, while we learn latent correlations among nodes with new connectivities and edge weights, showing the superiority of graph learning.
2) Compared with GLCN and GAT that learn only edge weights based on existing connectivities in graphs, we outperform by 3.1% and 2.3% on average. This is because we learn new connectivities that fully capture latent correlations between nodes.
3) Compared with AGCN that learns both edge weights and connectivities but is task driven with no data prior, we outperform by 2.0% on average. This is because the proposed cross-space Graph Laplacian Regularizer enforces smoothness of the input data with respect to the graph over the feature space and achieves similarity-preserving mapping on graphs, thus leading to better and more robust results.

===
Response to <Reviewer 1>:
[Q1] Analysis on why the new objective discloses similarity

We provide analysis in the spatial domain and spectral domain respectively below.
1) In the spatial domain, by minimizing the cross-space GLR as objective in Eq. 4, when two nodes are dissimilar in the input space, the learned edge weight tends to be small, enforcing the distance in the new feature space to be large. Therefore, GLR preserves the similarity between the new feature space and the input space.
2) From a graph spectral view, the GLR term can be rewritten as
x' L x = x' (U Λ U') x = (U' x)' Λ (U' x) = a' Λ a = Σ λ_i {a_i}^2,
where L is the Laplacian matrix, U and Λ denote eigenvectors and eigenvalues of L, a is the transform coefficient vector and \lambda_i is the i-th eigenvalue. Minimizing this term essentially penalizes high-frequency components (a_i w.r.t. large \lambda_i). Pairs of dissimilar nodes have larger variations, corresponding to high-frequency components, so minimizing GLR tends to learn smaller edge weights to penalize high-frequency components.

[Q2] How the similarity structure of graph changed after applying RGLN?

Fig. 5 of our submission shows the change in similarity structure of a graph:
1) If the provided graph is incomplete or unavailable as in Fig. 5(a), our proposed method learns more latent connectivities, as shown in Fig. 5 (b).
2) After applying RGLN, similar nodes keep similar while dissimilar nodes are pushed away. As in Fig. 5, the connectivities of three similar node pairs in Fig. 5(a) are still preserved in (b) (dark orange blocks), while many dissimilar node pairs are also captured in (b) (light yellow blocks).

===
Response to <Reviewer 2>:
[Q1] The datasets used are small in the paper, why not use DBLP dataset?

1) The size of our used Pubmed dataset (V=18230, E=79612, D=500, K=3) is comparable to DBLP dataset (V=17716, E=105734, D=1639, K=4). Besides, with our dense adjacency matrix and low-rank assumption in distance metric learning, the number of edges (E) and dimension of features (D) will have little influence on the performance. So we consider our experiments sufficient to support our idea.
2) We choose the most commonly used datasets mainly for fair comparison with other graph learning methods, since these methods haven’t used DBLP dataset.

[Q2] It is better to use dynamic graph in different timestamp instead of dropping out edges for missing ratio.

The experiment on missing edges aims to evaluate the robustness of the proposed method when the provided graph is incomplete (with missing edges), since robustness to missing edges is important in practice. We will consider extending our method to dynamic graph sequence learning.

===
Response to <Reviewer 3>:
[Q] May also cite and discuss some graph matching papers

We will cite and discuss the suggested relevant papers. These papers study the problem of similarity learning for graphs including cross-graph affinity learning as well as cross-graph embedding for topology recovery, which are related to our work in terms of learning similarities in graphs. On the other hand, while these papers focus on learning cross-graph similarity for matching, we focus on similarity learning inside a single graph to capture the data correlation. Also, while both our method and the suggested paper [3] predict missing edges to recover graph topology, we focus on prediction from one single graph.

-------------



Thanks for your valuable comments and suggestions.
Common questions:
\1. Novelty (Reviewer 1 & 2)
Though related, our model is not an integration of GLCN and AGCN.
1) We are the first to pose a joint learning problem of the underlying graphs and node features at EACH layer in GCNNs, which aims to optimize the entire network model.
2) GLCN assumes the edge connectivity in the graph is known and only learns a non-negative function as edge weights, while our method learns both edge connectivity and edge weights, which is more encompassing.
3) Different from the task-driven AGCN where the Mahalanobis distance metric M is learned by minimizing the cross-entropy only, our model is both task-driven and data-adaptive by optimizing M via both the GLR and cross-entropy. Also, instead of assuming a general M in AGCN, we assume M is low rank to reduce number of parameters.

\2. Purpose of GLR loss (Reviewer 1 & 3)
The motivation and analysis of using GLR loss is discussed in the second subsection of “Background in Spectral Graph Theory”. Specifically, minimizing GLR forces the graph to capture the similarity of the graph signal, thus optimizing the underlying graph.
We can further provide theoretical analysis from the frequency domain. That is, the GLR term x^T L x can also be written as the sum of the transform coefficients of x weighted by the eigenvalues of L. Since a larger eigenvalue corresponds to a higher-frequency transform coefficient, minimizing GLR essentially penalizes high-frequency components, thus leading to a low-pass and smooth graph signal with respect to the graph.

Reviewer 1
\1. Graph signal and features
Graph signals are data defined on vertices of a graph. In point clouds, if we treat each point as a node in a graph, the 3D coordinate of each point is the graph signal. In citation networks, the input features of each node are the graph signals. In general, graph signals are set as the input data/features on nodes in the experiments. In contrast, features in our context refer to latent features extracted from graph convolution.

\2. Point cloud classification and graph classification
Point cloud classification problem can be viewed as graph classification problem as we build a graph for each point cloud and classify point clouds based on graphs. We will make it clearer.

\3. Standard deviation
Since not all methods we compare in the paper provide standard deviation, we didn't present it in the table. For our experiments, the standard deviations for Citeseer, Cora and Pubmed are 0.0046, 0.0035, 0.0038 respectively.

\4. Complexity
Sparse implementations of GCN have time complexity of O(|E|FC) where |E| is the number of edges, F is the input feature dimension and C is the output feature dimension. Dense implementations of graph convolution operations including our method all have time complexity of O(N^2 FC), where N is the number of nodes. This work is a first attempt to jointly optimize the underlying graph and node features at each layer, with the reduction of computation complexity considered as future work.

Reviewer 2
\1. Difference with previous similarity learning approaches
Graph matching problems learn the similarity of nodes between different graphs while we learn the similarity of different nodes inside one graph, like the self-attention operation.

\2. Subsampling
For the Pubmed dataset, we simply choose the first 10000 nodes to evaluate all methods. The overall accuracy actually decreases compared to that of performing on the entire dataset, since fewer number of nodes are used for training, resulting in training data with less information.

\3. Clarification
We will clarify the aspects the reviewer suggested. To make it clear, this paper is related to the family of graph-based methods, because we focus on the joint learning of graphs and node features without assumptions on the input data. Besides, the recovery of graph structure from pairwise affinity is defined in Eq. 8.

\4. Graph learning
Thanks for the valuable comments. We choose a fully connected graph topology mainly for simplicity and effectiveness. In future, we will consider discretization / projection step to yield new connectivity. Though the idea of building a different graph across different layers was previously proposed in DynGraphCNN, they are not learning / optimizing a different graph across layers but perform empirical k-NN construction of graphs.

Reviewer 3
\1. Presentation
We will rephrase and properly discuss DGCNN and FeaStNet in the intro. To make it clear, DGCNN builds graphs based on the empirical k-NN method across different layers, while we explicitly learn / optimize the graph based on vertex features. FeaStNet applies metric learning only on local neighbors while we learn a global fully-connected graph. We will also rephrase the term “Joint”.

\2. Solve R
R is updated via back propagation and is solved within the network. We will compare the final energy between the network's solution and the global one.

\3. Eq. 11
Eq. 11 is improvement of GCN where $\tilde A=\Lambda^{-\frac 1 2}(A+I)\Lambda^{-\frac 1 2}$. Specifically, we replace the identity matrix $I$ (meaning adding self-loop) with a learned graph $A^*$ of the current layer for optimized graph representation. In our method, $\Lambda_{ii}=\sum_j(A_{ij}+A^*_{ij})$, i.e., the degree matrix computed from (A+A^*). Note that we omit subscripts here for simple notations.

\4. cross entropy loss
Actually, as shown in Fig. 2, we only employ the cross entropy loss to the last layer, and use graphs learned from all the layers to compute the GLR loss. We will make it clearer.

\5. Low-rank approximation of the mahalnobis matrix M
In the first layer, K is the dimension of input features (3703, 1433, 500 for Citeseer, Cora, Pubmed respectively), and is thus much larger than S=16. In the second layer, we choose S=K=16 without further reducing the feature dimension.