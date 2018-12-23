# Binary Tree

* Every Node has 5 forms: 

  NULL, only root, root + left child, root + right child, root + both children.

  > N nodes can make how many different BTs?
  >
  > Catalan Number：$\frac{1}{n+1}C_{2n}^n$
  >
  > How many edges does a BT with N nodes have? 
  >
  > $N-1$

  Yet another deduction of Catalan Number:
  $$
  f(0) = 1 \\ 
  f(1) = 1 \\
  f(2) = f(1)f(0) + f(0)f(1) = 2 \\
  f(3) = f(2)f(0) + f(1)f(1) + f(0)f(2) = 5 \\
  ... \\
  f(n) = f(n-1)f(0) + f(n-2)f(1) + ... + f(1)f(n-2) + f(n-1)f(0) \\
  the\ solution\ is \ f(n) = \frac{C_{2n}^n}{n+1}
  $$

  ```c++
  template <class T>
  struct node{
      node* left, right;
      T data;    
      node(){}
      node(const T& d) {data = d;}
  };
  
  template <class T>
  struct BinaryTree{
    	node* root;
      BinaryTree() {root=NULL;}
      void addLeft(node* r, const T& d);
      void addRight(node* r, const T& d);
  };
  ```

* Some Special forms

  * 满二叉树（Full BT）

    Every node's degree is either 0 or 2.

    (定义并不统一，在这里我们如此定义。)

  * 完全二叉树（Complete BT）

    * Definition:

      Only the nodes **in the last two layers** can have degrees less than 2.

      The last layer's nodes are aligned **continuously from left to right.**

    * Features

      Leaves only exist in the last two layers.

      **内部路径长度和**在具有相同节点数的二叉树中最短。（根节点到各节点的路径长度总和）

      可以连续的存储在顺序表中。

   * 扩充二叉树

      * 定义：把空指针全部替换为空树叶（外部节点）。

      * 性质：

        扩充二叉树是满二叉树。

        外部节点个数为内部节点数加一。（$N_0 = N_2 + 1$）

        **外部路径长度E，内部路径长度I，内部节点个数n，则$E= I + 2n$**

        归纳法证明：
        $$
        E_n = I_n + 2n \\
        assume\ add \ a\ node\ of\ length= k \\
        I_{n+1} = I_n + k \\
        E_{n+1} = E_n -k+2(k+1) = E_n+k+2 \\
        E_{n+1} = I_{n+1}+2(n+1)
        $$

* 二叉树的性质

  * （满二叉树定理）：非空满二叉树$N_0 = N_2 + 1$

  * （满二叉树定理推论）：非空二叉树空指针数$P = N + 1$，N为总结点数。

    $N(N>0)$个节点的$K$叉树的空指针数：
    $$
    P=NK - (N-1) = N(K-1)+1
    $$

  * 实际上，任意二叉树$N_0 = N_2 + 1$
    $$
    N = N_0 + N_1 + N_2  = E +1\\
    E = N_1 + 2N_2 \\
    \Rightarrow N_0 = N_2 + 1
    $$

  * 第$i$层（根节点规定为第0层）最多有$2^i$个节点。

  * 二叉树的**高度**即层数，**深度**即最长路径长度。

    （仅根节点的树高度为1，深度为0。）

    高度为$K$的二叉树最多有$2^K - 1$个节点。($\sum_{i=0}^{K-1}2^i$)

  * N个节点的完全二叉树高度为**$ceil(log_2(N+1))$**

    $2^{k-1}-1  \lt N \le 2^k-1$

    第$n$层的首元素为$2^n-1$ (从零开始编号)

  * N个节点的完全二叉树节点从上到下从左到右从零开始编号（数组存储）：

    * 根节点$i=0$，最后一个节点$i=n-1$
    * $i$的父节点$floor((i-1)/2)$
    * $i$的左子节点为$2i+1$，右子节点$2i+2$ （大于$n-1$则不存在）
    * $i$为偶数时左兄弟为$i-1$，$i$为奇数时右兄弟为$i+1$ （不在范围内则不存在）

* 二叉树的周游（Traversal）

  * DFS (Recursion)

    * Prefix: Root --> Left --> Right
    * Infix: Left --> Root --> Right
    * Postfix: Left --> Right --> Root

    重构算法性质：**先序或后序+中序**可以确定一棵二叉树，先序和后序无法确定。

  * 非递归DFS

    手动用循环解决递归。

    ```c++
    // prefix
    void prefix(node* root){
        stack<node<T>*> stk;
        node* p = root;
        stk.push(NULL); // end condition for while.
        while(p){
            visit(p);
            if(p->right!=NULL) stk.push(p->right);
            if(p->left!=NULL) p = p->left;
            else p = stk.top(), stk.pop();
        }
    }
    
    // infix
    void infix(node* root){
        stack<node<T>*> stk;
        node* p = root;
        while(!stk.empty() || p){
            if(p){
                stk.push(p);
                p = p->left;
            }
            else{
                p = stk.top(), stk.pop();
                visit(p);
                p = p->right;
            }
        }
    }
    
    // postfix
    // slightly complicated, for we need to additional tags.
    enum Tags{left, right};
    template <class T>
    struct stackElement{
    	node<T>* p;
        Tags tag;
    };
    void postfix(node* root){
        stack<stackElement<T>*> stk;
        stackElement<T> ele;
        node* p = root;
        while(!stk.empty() || p){
            if(p!=NULL){
                ele.p = p;
                ele.tag = left;
                stk.push(ele);
                p = p->left;
            }
            else{
                ele = stk.top(), stk.pop();
                if(ele.tag == left){
                    ele.tag = right;
                    stk.push(ele);
                    p = p->right;
                }
                else{
                    visit(p);
                    p = NULL; // continue popping
                }
            }
        }
    }
    ```

  * BFS

  ```c++
  //My implementation of a toy tree.
  #include <iostream>
  #include <algorithm>
  #include <string>
  #include <stack>
  #include <queue>
  #include <set>
  using namespace std;
  
  struct node {
  	node *l, *r;
  	char data;
  };
  
  string res;
  
  void visit(node* n) { res += n->data; }
  
  void prefix(node* root) {
  	if (root != NULL) {
  		visit(root);
  		prefix(root->l);
  		prefix(root->r);
  	}
  }
  
  void prefix2(node* root) {
  	stack<node*> stk;
  	node* n = root;
  	while (n) {
  		visit(n);
  		if(n->r != NULL) stk.push(n->r);
  		if(n->l != NULL) stk.push(n->l);
  		if (stk.empty()) return;
  		n = stk.top(); stk.pop();
  	}
  }
  
  void infix(node* root) {
  	if (root != NULL) {
  		infix(root->l);
  		visit(root);
  		infix(root->r);
  	}
  }
  
  
  void infix2(node* root) {
  	stack<node*> stk;
  	set<node*> vis;
  	node* n = root;
  	while (n) {
  		if (n->l != NULL && !vis.count(n->l)) {
  			if (n->r != NULL) stk.push(n->r);
  			stk.push(n);
  			n = n->l;
  		}
  		else {
  			if (n->r != NULL && !vis.count(n->l)) stk.push(n->r); // null left sub tree
  			visit(n);
  			vis.insert(n);
  			if (stk.empty()) return;
  			n = stk.top(); stk.pop();
  		}
  	}
  }
  
  void postfix(node* root) {
  	if (root != NULL) {
  		postfix(root->l);
  		postfix(root->r);
  		visit(root);
  	}
  }
  
  void postfix2(node* root) {
  	stack<node*> stk;
  	set<node*> vis;
  	node* n = root;
  	while (n) {
          // left first
  		if (n->l != NULL && !vis.count(n->l)) {
  			stk.push(n);
  			if (n->r != NULL) stk.push(n->r);
  			n = n->l;
  		}
          // right second
  		else if (n->r != NULL && !vis.count(n->r)) {
  			stk.push(n);
  			n = n->r;
  		}
          // finally root
  		else {
  			visit(n);
  			vis.insert(n);
  			if (stk.empty()) return;
  			n = stk.top(); stk.pop();
  		}
  	}
  }
  
  // NB! Universal DFS algorithm.
  enum tag{Left, Right};
  struct element {
  	node* n;
  	tag t;
  };
  void dfs(node* root, int order) {
  	stack<element> stk;
  	element ele;
  	node* n = root;
  	while (!stk.empty() || n) {
  		while (n != NULL) {
  			if (order == 0) visit(n);  //prefix
  			ele.n = n;
  			ele.t = Left;
  			stk.push(ele);
  			n = n->l;
  		}
  		ele = stk.top(); stk.pop();
  		n = ele.n;
  		if (ele.t == Left) {
  			if (order == 1) visit(n);  //infix
  			ele.t = Right;
  			stk.push(ele);
  			n = n->r;
  		}
  		else {
  			if (order == 2) visit(n);  //postfix
  			n = NULL;
  		}
  	}
  }
  
  void bfs(node* root) {
  	queue<node*> que;
  	que.push(root);
  	while (!que.empty()) {
  		node* cur = que.front(); que.pop();
  		visit(cur);
  		if (cur->l != NULL) que.push(cur->l);
  		if (cur->r != NULL) que.push(cur->r);
  	}
  }
  
  void preRebuild(node*& root, string infix, string prefix) {
  	int len = infix.length();
  	if (len == 0) return;  // null left sub tree
  	char r = prefix[0];
  	int pos = infix.find(r);
  	if (root == NULL) root = new node();
  	root->data = r;
  	if (len == 1) return; // leaves
  	preRebuild(root->l, infix.substr(0, pos), prefix.substr(1, pos));
  	preRebuild(root->r, infix.substr(pos + 1, len - pos - 1), prefix.substr(pos + 1, len - pos - 1));
  }
  
  void postRebuild(node*& root, string infix, string postfix) {
  	int len = infix.length();
  	if (len == 0) return;  // null left sub tree
  	char r = postfix.back();
  	int pos = infix.find(r);
  	if (root == NULL) root = new node();
  	root->data = r;
  	if (len == 1) return; // leaves
  	postRebuild(root->l, infix.substr(0, pos), postfix.substr(0, pos));
  	postRebuild(root->r, infix.substr(pos + 1, len - pos - 1), postfix.substr(pos, len - pos - 1));
  }
  
  void del(node* root) {
  	if (root != NULL) {
  		del(root->l);
  		del(root->r);
  		delete root;
  	}
  }
  
  int main() {
  	node* rt = NULL;
  	preRebuild(rt, "DGFHBAICME", "ABDFGHCIEM");
  	res = "";
  	postfix(rt);
  	cout << res << endl;
  	res = "";
  	postfix2(rt);
  	cout << res << endl;
  	res = "";
  	bfs(rt);
  	cout << res << endl;
  }
  ```

* Storage

  * Dynamic Bifurcated Linked List
  * Static Sequential List ( **(nearly) Complete Binary Tree**)

* Binary Search Tree (BST)  **二叉排序/搜索树**

  每个节点K的左子树的所有节点小于K，右子树的所有节点大于K。

  性质：节点的值唯一；中序遍历有序。

  最佳插入/删除/查找时间复杂度$O(logn)$

  ```c++
  struct node {
  	node* L, R;
      int data;
  };
  
  // node* root = build(NULL, data[0])
  node* build(node* root, int d){
      root = new node();
      root->data = d;
      return root;
  }
  
  void insert(node* root, int d){
      if(root->data == d) return;
      else if (root->data < d){
          if(root->R == NULL){
              root->R = build(root->R, d);
              return;
          }
          else insert(root->R, d);
      }
      else{
          if(root->L == NULL){
              root->L = build(root->L, d);
              return;
          }
          else insert(root->L, d);
      }
  }
  
  node* parent(node* root, node* p){
      if(root == p) return NULL;
      else if(root->L == p || root->R == p) return root;
      else if(root->L != NULL) return parent(root->L, p);
      else if(root->R != NULL) return parent(root->R, p);
      else return NULL;
  }
  
  // non-recursive
  void del(node* root, node* p){
      node* fp = parent(root, p);
      // left sub tree is empty: naive
      if(p->L == NULL){
          if(fp->L == p) fp->L = p->R;
          else fp->R = p->R;
      }
      // find left subtree's largest value, replace p with it and delete p.
      // or find right subtree's smallest value, replace p and delete.
      else{
          node* lmax = p->L;
          node* lpar = p;
          while(lmax->R != NULL) {
              lpar = lmax;
              lmax = lmax->R;
          }
          lpar->R = NULL;
          lmax->L = p->L;
          lmax->R = p->R;
          if(fp->L == p) fp->L = lmax;
          else fp->R = lmax;
      }
      delete p;
  }
  ```

  定理：**随机构造$n$个不同节点的二叉搜索树的平均深度为$O(log\ n)$，期望内部路径总和为$O(nlog\ n)$**
  $$
  D(N)\ is \ inner \ path \ summation.\\
  D(N) = D(i) + D(N-1-i) + N-1 \\
  D(N) = 2*\frac 1 N\sum_{i=0}^{N-1}D(i) + N-1 \\
  ND(N)-(N-1)D(N-1) = 2D(N-1) + 2(N-1) \\
  \frac {D(N)} {N+1} = \frac {D(N-1)}{N} + \frac{2(N-1)}{N(N+1)} \\
  f(n) = f(n-1)+O(1/n) \\
  f(n) \in \sum_{i=1}^nO(1/n) = O(log\ n)
  $$
  $f(n)$相当于平均高度，从而期望内部路径总长$D(N)=O(NlogN)$

  **Harmonic Series：**
  $$
  \sum_{i=1}^{N}\frac 1 i \in O(logN) \\
  this \ follows \ continuous \ condition: \\
  \int \frac 1 xdx = lnx
  $$
  Another version of proof:

  ![img](https://i.stack.imgur.com/xHenH.png)

* Heap 堆

  最大值堆：每一个节点的值K大于其左右节点的值。（**局部有序**，不同于BST的全局有序。）

  最小值堆：每一个节点的值K小于其左右节点的值。

  对同一组数据，可以构建许多不同的堆。

  一般存储为静态的完全二叉树数组。

  ```c++
  const int maxN = 1000;
  int N;
  int arr[maxN];
  
  // sink i until it is smaller than its children
  void siftdown(int i){
      int tmp = arr[i];
      int j = 2*i+1;
      while(j<N){
          if( j<N-1 && arr[j]>arr[j+1]) j++;
          if(tmp>arr[j]){
              arr[i] = arr[j];
              i = j;
              j = 2*j+1;
          }
          else break;
      }
      arr[i] = tmp;
  }
  
  
  // lift i until its parent is smaller than it (or to top).
  void siftup(int i){
      int tmp = arr[i];
      while(i>0 && arr[(i-1)/2]>tmp){
          arr[i] = arr[(i-1)/2];
          i = (i-1)/2;
      }
      arr[i] = tmp;
  }
  
  
  // change arr into a min heap, O(n)
  void build(int root){
      // N/2-1 is the last father.
      for(int i = N/2 - 1; i>=0; i--){
          siftdown(i);
      }
  }
  
  // O(logn)
  bool insert(int d){
      if(N == maxN) return false;
      arr[N] = d;  // add new data to the bottom
      siftup(N);
      N++;
      return true;
  }
  
  // O(logn)
  int pop(){
      if(N==0) return -1;
      swap(arr[0], arr[--N]); // swap to bottum (out of heap)
      if(N>1) siftdown(0);
      return arr[N]; 
  }
  
  // remove position i ans assign deleted val to node, O(logn)
  bool remove(int i, int& node){
      if(i<0 || i>=N) return false;
      int tmp = arr[i];
      arr[i] = arr[--N];
      siftup(i);
      siftdown(i);
      node = tmp;
      return true;
  }
  ```

  证明sift建堆时间复杂度最坏$O(n)$：

  用到了**错位相减法**求数列和。
  $$
  \sum_{i=0}^{logn}2^i(logn - i) = \sum_{j=0}^{logn}n\frac j {2^j} \lt 2n
  $$
  有趣的题目：

  ![1541493860686](E:\aa\junior1\DSAlgo\DA05_BinaryTree.assets\1541493860686.png)

* Priority Queue

  堆是优先队列的一种实现方式。

* Huffman Tree (最优二叉树)

  * 定义：具有最小带权路径长度的二叉树。一定是**满二叉树**。

    **Huffman的外部路径长度**一般指的是带权路径长度，不要忘了叶子节点的权重。

    应用：频率不等的字符，可以用出现频率来编码，Huffman提供最优的不等长编码方式。

  * 前缀编码方式：每棵子树左枝标0，右枝标1，从根到叶连接即该叶子的编码。**任何一个字符的编码都不是另一个字符编码的前缀（反编码唯一）。**

  * K叉Huffman树：

    直接使用2叉方法可能导致第一层不满，故应该添加**虚叶子结点（权重为零）**。

  ```c++
  #define  _CRT_SECURE_NO_WARNINGS
  #include <iostream>
  #include <algorithm>
  #include <string>
  #include <stack>
  #include <vector>
  #include <queue>
  #include <set>
  using namespace std;
  
  const int K = 2;
  
  struct node {
  	node* children[K];
  	int w;
  	node(int w) :w(w) {
  		for (int i = 0; i < K; i++) children[i] = NULL;
  	}
  	node(int w, node** _children) :w(w) {
  		for (int i = 0; i < K; i++) children[i] = _children[i];
  	}
  	bool operator<(const node& b) const{
  		return w < b.w;
  	}
  };
  
  vector<node*> tree;
  node* build(vector<int>& weights) {
  	tree.clear();
  	int N = weights.size();
      // add virtual nodes
  	for (int i = 0; i < K - 1 - (N - 1) % (K - 1); i++) weights.push_back(0);
  	priority_queue<node*> q;
  	for (int w : weights) {
  
  		tree.push_back(new node(w));
  		q.push(tree.back());
  	}
  	while (true) {
  		node* vs[K];
  		int sum = 0;
  		for (int i = 0; i < K; i++) {
  			vs[i] = q.top();
  			sum += vs[i]->w;
  			q.pop();
  		}
  		tree.push_back(new node(sum, vs));
  		if (q.empty()) return tree.back();
  		q.push(tree.back());
  	}
  }
  
  int res = 0;
  // weighted route length
  void WRL(node* root, int depth) {
  	bool nil = true;
  	for (int i = 0; i < K; i++) {
  		if (root->children[i] != NULL) {
  			WRL(root->children[i], depth + 1);
  			nil = false;
  		}
  	}
  	if (nil) res += depth * root->w;
  }
  
  // copied code for K=2 only
  int _print_t(node *tree, int is_l, int offset, int depth, char s[20][255])
  {
  	char b[20];
  	int width = 5;
  
  	if (!tree) return 0;
  
  	sprintf(b, "(%03d)", tree->w);
  
  	int l = _print_t(tree->children[0], 1, offset, depth + 1, s);
  	int r = _print_t(tree->children[1], 0, offset + l + width, depth + 1, s);
  
  	for (int i = 0; i < width; i++)
  		s[depth][offset + l + i] = b[i];
  
  	if (depth && is_l) {
  
  		for (int i = 0; i < width + r; i++)
  			s[depth - 1][offset + l + width / 2 + i] = '-';
  
  		s[depth - 1][offset + l + width / 2] = '.';
  
  	}
  	else if (depth && !is_l) {
  
  		for (int i = 0; i < l + width; i++)
  			s[depth - 1][offset - width / 2 + i] = '-';
  
  		s[depth - 1][offset + l + width / 2] = '.';
  	}
  	return l + width + r;
  }
  
  void print_t(node *tree)
  {
  	char s[20][255];
  	for (int i = 0; i < 20; i++)
  		sprintf(s[i], "%80s", " ");
  
  	_print_t(tree, 0, 0, 0, s);
  
  	for (int i = 0; i < 20; i++)
  		printf("%s\n", s[i]);
  }
  
  int main() {
  	vector<int> weights{ 1,4,9,6,8,10,12 };
  	node* root = build(weights);
  	print_t(root);
  	res = 0;
  	WRL(root, 0);
  	cout << res << endl;
  }
  ```
