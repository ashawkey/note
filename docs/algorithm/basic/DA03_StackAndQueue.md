# Stack and Queue

### Stack (LIFO)

* **Catalan Number**

  $\frac{1}{n+1}C_{2n}^n$

  * **推导：**

  假设进栈和出栈分别用1和0代表，则n个火车的出入站组合可以表示为一个2n的序列，其中有n个0和n个1（包含不合法的序列）。这样的序列一共有$C_{2n}^n$种。

  考虑其中不合法的序列，一定满足从左到右扫描到某一位（假设为第2m+1位）时出现了m个1和m+1个0（出栈次数多于入栈次数）。此时这个序列在2m+2位以后还有n-m个1，n-m-1个0。把此后的1与0互换，则任意不合法的序列可以唯一对应到另一个2n序列，且对应的序列满足有n+1个0，n-1个1。即这样的序列一共有$C_{2n}^{n-1}$个。

  所以合法的序列一共有$C_{2n}^n - C_{2n}^{n-1} = \frac{C_{2n}^n}{n+1}$个。

  * **性质**：

  
$$
\displaylines{
C_0 = 1 \ \  and \ \ C_{n+1} = \sum_{i=0}^nC_iC_{n-i}  \\
  or\ \  C_{n+1} = \frac{2(2n+1)}{n+2} C_n
}
$$


* Storage

  * Sequential (more often used)
  * Linked List

* Recursion

  * Concept: **a function calls itself explicitly or implicitly.**

  * Implement

    * Recursive Formula
    * End Condition

  * 函数运行的存储分配

    * 静态：非递归，调用与返回处理比较简单。

    * 动态：递归，在内存中开辟足够大的动态区（运行栈）。

      动态存储区包括堆（非LIFO数据，如指针分配）与栈（LIFO数据，如函数调用）。

  * 递归程序的非递归化

    一般使用`for loop`代替。

### Expression Evaluation

* Type

  **infix, prefix, postfix**

* Recursive CFG definition 

* Postfix expression evaluation

  No parenthesis in postfix expression.

  Easy for computer to evaluate. (a single stack is enough.)

* Evaluate Infix 

  ```c++
  #include <iostream>
  #include <cstring>
  #include <algorithm>
  #include <queue>
  #include <string>
  #include <vector>
  #include <stack>
  
  using namespace std;
  int N;
  string e;
  
  int prior(char c) {
  	if (c == '+' || c=='-') return 1;
  	else if (c == '*' || c=='/') return 2;
  	else if (c == '(') return 0;
  }
  
  void popcalc(stack<char>& ops, stack<int>& nums) {
  	char op = ops.top(); ops.pop();
  	int b = nums.top(); nums.pop();
  	int a = nums.top(); nums.pop();
  	int res = 0;
  	if (op == '+') res = a + b;
  	else if (op == '-') res = a - b;
  	else if (op == '*') res = a * b;
  	else if (op == '/') res = a / b;
  	nums.push(res);
  }
  
  int eval(string e) {
  	stack<char> ops;
  	stack<int> nums;
  	for (int i = 0; i < e.size(); i++) {
  		if (isdigit(e[i])) {
  			int num = 0;
  			int j = i;
  			for (j; j < e.size(); j++) {
  				if (isdigit(e[j])) num = num * 10 + e[j] - '0';
  				else break;
  			}
  			i = j - 1;
  			nums.push(num);
  		}
  		else if (e[i] == '(') ops.push(e[i]);
  		else if (e[i] == ')') {
  			while (ops.top() != '(') popcalc(ops, nums);
  			ops.pop();
  		}
  		else {
  			while (!ops.empty() && prior(ops.top()) >= prior(e[i])) popcalc(ops, nums);
  			ops.push(e[i]);
  		}
  	}
  	while (!ops.empty()) popcalc(ops, nums);
  	return nums.top();
  }
  
  int main() {
  	cin >> N;
  	for (int i = 0; i < N; i++) {
  		cin >> e;
  		cout << eval(e) << endl;
  	}
  }
  ```


### Queue (FIFO)

* Implement

  * sequential list

    **front**: real pointer.

    **rear**: imaginary pointer.

  * linked list

    both pointer point to real number.

* Sequential List ：Overflow

  多申请一个空间，并使rear指针指向虚的队尾：用来**区分队空与队满。**

  * 假溢出：`rear == mSize-1`但队首还有很多空余位置。需要使用循环队列解决。

  * 循环队列：

    ```c++
    struct myqueue{
    	int mSize, front, rear;
    	T *que;
    	myqueue(int size)  {
            mSize = size + 1;
            que = new T[mSize];
            front = rear = 0;
    	}
    	bool push(const T item){
            if((rear+1)%mSize==front){
                return false;  // full
            }
            que[rear] = item;
            rear = (rear+1)%mSize;
            return true;
    	}
        bool pop(){
            if(front==rear){
                return false; // empty
            }
            front = (front + 1)%mSize;
            return true;
        }
    };
    ```

* Linked List：
  * 不存在溢出（队满）问题。
  * `rear == front`代表队空。
* Application
  * BFS
  * buffer


# Simulation

* 2 stacks --> queue

* 2 queue --> stack

* Better Solution

  ```
  //questack
  SPUSH(x):
  	PUSH(Q1, x)
  
  SPOP(x):
  	if empty(Q1):
  		while not empty(Q2):
  			POP(Q2, y)
  			PUSH(Q1, y)	
  	while not empty(Q1):
  		POP(Q1, y)
  		if empty(Q1):
  			x = y
  		else:
  			PUSH(Q2, y)
  
  Sempty():
  	return empty(Q1)&&empty(Q2)
  
  
  // stkqueue
  // O(.) is the same, but empty() called less. And more concise.
  QPUSH(x):
  	PUSH(S1, x)
  	
  QPOP(x):
  	if not empty(S2):
  		POP(S2, x)
  	else:
  		while not empty(S1):
  			POP(S1, y)
  			PUSH(S2, y)
  		POP(S2, x)
  		
  Qempty():
  	return empty(S1) && empty(S2)
  ```