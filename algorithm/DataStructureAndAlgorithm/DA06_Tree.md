# Tree

### Definition

* 有序树：

  子节点从左到右依次编号。

  **度为2的有序树并不是二叉树！**

  （第一子节点删除后第二子节点会顶替为第一子节点，二叉树则必须严格区分左右子节点）

* 森林：

  零棵或多棵不相交的树的集合（通常有序）。

  **添加一个根节点则成为一棵树。**

  树中的每个结点的子树组成一个森林。

* 森林（或树）与二叉树一一对应。

  **树对应的二叉树根节点没有右子树。**

### Traversal

* 先根：先根周游树对应于前序周游对应二叉树
* 后根：先根周游树对应于**中序**周游对应二叉树

* BFS

### Storage

* 链式存储

  * 子节点表示法：难以找兄弟。

  * 动态指针数组：存储空间是动态的。

    最容易写。。。

  * **左孩子右兄弟表示法（二叉链表）**

    本质上存储了对应的二叉树。

    ```c++
    #include <iostream>
    #include <queue>
    
    using namespace std;
    
    struct node {
    	node *lc, *rs; // left child, right sibling
    	int d;
    };
    
    void build(node*& rt, int d) {
    	rt = new node();
    	rt->d = d;
    }
    
    void insert(node* n, int d) {
    	if (n->lc == NULL) {
    		n->lc = new node();
    		n->lc->d = d;
    		return;
    	}
    	else {
    		node* tmp = n->lc;
    		while (tmp->rs != NULL) tmp = tmp->rs;
    		tmp->rs = new node();
    		tmp->rs->d = d;
    		return;
    	}
    }
    
    bool isleaf(node* t) {
    	return t->lc == NULL;
    }
    
    node* parent(node* rt, node* t) {
    	queue<node*> q;
    	q.push(rt);
    	while (!q.empty()) {
    		node* p = q.front(); q.pop();
    		node* c = p->lc;
    		while (c != NULL) {
    			if (c == t) return p;
    			q.push(c);
    			c = c->rs;
    		}
    	}
    	return NULL;
    }
    
    node* leftsib(node* rt, node* t) {
    	node* p = parent(rt, t)->lc;
    	while (p->rs != NULL){
    		if (p->rs == t) return p;
    		p = p->rs;
    	}
    	return NULL;
    }
    
    // delete in binary tree mode
    void _delsub(node* n) {  
    	if (n != NULL) {
    		_delsub(n->lc);
    		_delsub(n->rs);
    		delete n;
    	}
    }
    
    // delete in tree mode
    void delsub(node* rt, node* n) {
    	node* ls = leftsib(rt, n);
    	if (ls == NULL) {  // first child of parent, need to modify parent links
    		node* p = parent(rt, n);
    		if (p != NULL) {
    			p->lc = n->rs;
    			n->rs = NULL;
    		}
    	}
    	else{  // just cut this subtree out
    		ls->rs = n->rs;
    		n->rs = NULL;
    	}
    	_delsub(n); // delete binary tree
    }
    
    void visit(node* n) {
    	cout << "visit " << n->d << endl;
    }
    
    void prefix(node* rt) {
    	while (rt != NULL) {
    		visit(rt);
    		prefix(rt->lc);
    		rt = rt->rs;
    	}
    }
    
    void prefix2(node* rt) {
    	if (rt != NULL) {
    		visit(rt);
    		prefix(rt->lc);
    		prefix(rt->rs);
    	}
    }
    
    void postfix(node* rt) {
    	while (rt != NULL) {
    		postfix(rt->lc);
    		visit(rt);
    		rt = rt->rs;
    	}
    }
    
    void postfix2(node* rt) {
    	if (rt != NULL) {
    		postfix2(rt->lc);
    		visit(rt);
    		postfix2(rt->rs);
    	}
    }
    
    void bfs(node* rt) {
    	queue<node*> q;
    	q.push(rt);
    	while (!q.empty()) {
    		node* p = q.front(); q.pop();
    		visit(p);
    		p = p->lc;
    		while (p != NULL) {
    			q.push(p);
    			p = p->rs;
    		}
    	}
    }
    
    node* getchild(node* rt, int n) {
    	if (n == 0) return rt->lc;
    	node* tmp = rt->lc;
    	while (tmp != NULL && n--) {
    		tmp = tmp->rs;
    	}
    	return tmp;
    }
    
    int main() {
    	node* root = NULL;
    	build(root, 0);
    	insert(root, 1);
    	insert(root, 2);
    	insert(root, 3);
    	node* tmp = root->lc;
    	insert(tmp, 4);
    	tmp = getchild(root, 2);
    	insert(tmp, 5);
    	insert(tmp, 6);
    	cout << "===prefix===" << endl;
    	prefix(root);
    	cout << "===prefix2===" << endl;
    	prefix2(root);
    	cout << "===postfix===" << endl;
    	postfix(root);
    	cout << "===postfix2===" << endl;
    	postfix2(root);
    	cout << "===bfs===" << endl;
    	bfs(root);
    	cout << "===delete===" << endl;
    	delsub(root, getchild(root, 1));
    	cout << "===bfs===" << endl;
    	bfs(root);
    }
    ```

  * 父指针表示法（并查集）

    只存储父指针，应用于只需要知道每个节点与父节点的关系的情况。*父指针表示法可以唯一标识一棵无序树。*

    **标准重量权衡合并**：使得合并操作尽量平衡，保持$O(logn)$复杂度。

    ​	最简单的实现是每次把矮树合并到高树。

    **路径压缩**：每次合并都调整父节点到根节点，使得下一次检索仅需一步。

    ​	注意所有查询父节点路径上的经过点都要被压缩。

* 顺序存储

  主要关注**如何从线性序列还原树的链式结构**。

  （不考虑如何修改线性结构、如何将链式存储转换为顺序存储）

  * 带右链的先根表示法

    `rlink, info, ltag` naive.

  * 双标记的**先根表示法**

    `rtag, info, ltag`栈特性。

    一般0代表有，1代表没有。

    ```c++
    int N;
    int info[maxn];
    bool ltag[maxn];
    bool rtag[maxn];
    
    void tag2load() {
    	/*
    	本质先根搜索，rtag==0的节点与ltag==1的节点一一对应（除了最后一个节点ltag==1）。
    	r==0 && l==1 -> rs is the next node.
    	*/
    	stack<node*> stk;
    	node* p = new node();
    	stk.push(p);
    	for (int i = 0; i < N - 1; i++) {
    		p->d = info[i];
    		if (rtag[i] == 0) stk.push(p);
    		else p->rs = NULL;
    		node* np = new node();
    		if (ltag[i] == 0) p->lc = np;
    		else {
    			p->lc = NULL;
    			p = stk.top(); stk.pop();
    			p->rs = np;
    		}
    		p = np;
    	}
    	p->d = info[N - 1];
    	p->lc = p->rs = NULL;
    }
    ```

  * 带度数的**后根表示法**

    `info, degree`

    其他次序与度数结合无法重建树！

    适合于**动态指针数组**的存储方法，直观快捷。

  * 双标记的**层次表示法**

    `ltag, info, rtag`队列特性。

    ```c++
    void bfsload() {
    	queue<node*> stk;
    	node* p = new node();
    	stk.push(p);
    	for (int i = 0; i < N - 1; i++) {
    		p->d = info[i];
    		if (ltag[i] == 0) stk.push(p);
    		else p->lc = NULL;
    		node* np = new node();
    		if (rtag[i] == 0) p->rs = np;
    		else {
    			p->rs = NULL;
    			p = stk.front(); stk.pop();
    			p->rs = np;
    		}
    		p = np;
    	}
    	p->d = info[N - 1];
    	p->lc = p->rs = NULL;
    }
    
    ```

### K-ary Tree

子节点有序，类似二叉树。

