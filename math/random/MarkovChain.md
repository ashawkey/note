# Markov Chain

**Def.** 设$S$可数集，$\{X_n:n\ge 0\}$ 取值于$S$，若$\forall n\ge1, i_0, i_1, ..., i_{n-1}, i, j \in S$, $P(X_{n+1}=j|X_n=i, X_{n-1}=i_{n-1}, ..., X_0=i_0) = P(X_{n+1}=j|X_n=i)$, 则称$\{X_n:n\ge 0\}$为离散参数马尔科夫链。若$\forall n\ge0, i, j \in S, P(X_{n+1}=j|X_n=i)=P_{ij}$不依赖于n，则称$\{X_n:n\ge 0\}$为时齐的。

