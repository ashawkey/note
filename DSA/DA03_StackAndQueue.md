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
  C_0 = 1 \ \  and \ \ C_{n+1} = \sum_{i=0}^nC_iC_{n-i}  \\
  or\ \  C_{n+1} = \frac{2(2n+1)}{n+2} C_n
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
  /***************************************
   * Infix Expression Calculator
   *  input: 2 3 * 1 + ( 1 0 - 2 ^ 3 ) / 2
   *  ouput: 24
   * ************************************/
  #include <iostream>
  #include<cstdio>
  #include<algorithm>
  #include <cstring>
  #include<stack>
  #include <string>
  #include<math.h>
  #define MAX 210
  using namespace  std;
  
  int domath(char op0, int op1, int op2)
  {
  	if (op0 == '*')return op1 * op2;
  	else if (op0 == '/')return op1 / op2;
  	else if (op0 == '+')return op1 + op2;
  	else if (op0 == '-')return op1 - op2;
  	else return pow(op1, op2);
  }
  int getpriority(char a)
  {
  	int priority;
  	if (a == '^')priority = 3;
  	else if (a == '*' || a == '/')priority = 2;
  	else if (a == '+' || a == '-')priority = 1;
  	else if (a == '(')priority = 0;
  	return priority;
  }
  void popcalc(stack<int>& nums, char token) {
  	int operand2, operand1, result;
  	operand2 = nums.top();
  	nums.pop();
  	operand1 = nums.top();
  	nums.pop();
  	result = domath(token, operand1, operand2);
  	nums.push(result);
  }
  
  int main()
  {
  	string s;
  	getline(cin, s);
  	int n = s.size();
  	stack<int>nums;
  	stack<char>op;
  	for (int i = 0; i < n; ++i)
  	{
  		if (s[i] == ' ') continue;
  		if (isdigit(s[i])) {
  			string tmp;
  			int j = i;
  			for (j ; j < n; ++j) {
  				if (s[j] == ' ') continue;
  				if (!isdigit(s[j])) break;
  				else tmp += s[j];
  			}
  			i = j - 1;
  			nums.push(stoi(tmp));
  		}
  		else if (s[i] == '(')
  			op.push(s[i]);
  		else if (s[i] == ')')
  		{
  			char token = op.top();
  			op.pop();
  			while (token != '(')
  			{
  				popcalc(nums, token);
  				token = op.top();
  				op.pop();
  			}
  		}
  		else
  		{
  			while (!op.empty() && (op.top() != '(') && (s[i] != ' ') && (getpriority(op.top()) >= getpriority(s[i])))
  			{
  				char token = op.top();
  				op.pop();
  				popcalc(nums, token);
  			}
  			op.push(s[i]);
  		}
  	}
  	while (!op.empty())
  	{
  		char token = op.top();
  		op.pop();
  		popcalc(nums, token);
  	}
  	int res = int(nums.top());
  	cout << res << endl;
  	return 0;
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