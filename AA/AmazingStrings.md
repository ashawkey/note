# Amazing String Algorithms

### 最小覆盖子串

求最小覆盖子串。

* KMP

  **定理：最小覆盖子串长度为`N-next[N]`。**

  最小覆盖子串即前N-Next[N]个字符构成的子串。

#### Variants

* 二维最小覆盖矩阵面积（挤奶网络）

  愚蠢的char**传入后无法二维下标访问，所以只能写两套函数。

  ```c++
  #include <iostream>
  #include <cstring>
  #include <string>
  #include <algorithm>
  
  using namespace std;
  
  const int maxr = 10005;
  const int maxc = 80;
  int R, C;
  char mat[maxr][maxc];
  char tmat[maxc][maxr];
  
  bool coleq(int i, int j) {
  	for (int k = 0; k < R; k++)
  		if (mat[k][i] != mat[k][j]) return false;
  	return true;
  }
  
  bool roweq(int i, int j) {
  	for (int k = 0; k < C; k++)
  		if (tmat[k][i] != tmat[k][j]) return false;
  	return true;
  }
  
  int colmncov() {
  	vector<int> next(C + 1, 0);
  	next[0] = -1;
  	int i = 0, k = -1;
  	while (i < C) {
  		while (k >= 0 && !coleq(i, k)) k = next[k];
  		i++, k++;
  		next[i] = k;
  	}
  	return C - next[C];
  }
  
  int rowmncov() {
  	vector<int> next(R + 1, 0);
  	next[0] = -1;
  	int i = 0, k = -1;
  	while (i < R) {
  		while (k >= 0 && !roweq(i, k)) k = next[k];
  		i++, k++;
  		next[i] = k;
  	}
  	return R - next[R];
  }
  
  int main() {
  	cin >> R >> C;
  	memset(mat, '\0', sizeof(mat));
  	memset(tmat, '\0', sizeof(tmat));
  	for (int i = 0; i < R; i++) {
  		for (int j = 0; j < C; j++) {
  			cin >> mat[i][j];
  			tmat[j][i] = mat[i][j];
  		}
  	}
  	cout << rowmncov() * colmncov() << endl;
  }
  ```




### 最小重复子串

判断是否存在最小重复子串。

* KMP based

  ```c++
  class Solution {
  public:
      bool repeatedSubstringPattern(string s) {
          int len = s.size();
          s += "$";
          vector<int> next(len+1, 0);
          next[0] = -1;
          int i=0, k=-1;
          while(i<len){
              while(k>=0 && s[i] != s[k]) k = next[k];
              i++, k++;
              next[i] = k;
          }
          // minimum cover substring
          int cov = len - next[len];
          return cov<len && len%cov==0;
      }
  };
  ```

* Trick

  Trick is trick because you will never know why trick is trick.

  ```c++
  class Solution {
  public:
      bool repeatedSubstringPattern(string s) {
          return (s+s).substr(1,2*s.size()-2).find(s)!=-1;
      }
  };
  ```


### 最长回文子串

* Manacher



### 最长公共子串

求N个字符串的最长公共子串（每个字符串最长为L）

* KMP + Binary Search，$O(LlgL*N^2)$
* Suffix Array + Binary Search，$O(lgL*N^2)$



### 最长公共子序列

求2个字符串的最长公共子序列。

* DP



### 最长上升子序列

* DP



### Distinct Subsequences

求一个字符串的Unique的子序列的个数。

* DP

