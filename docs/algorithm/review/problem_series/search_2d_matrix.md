### [二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

 

并不是二维二分！没有二维二分法！

```
[[1, 3, 5],
 [2, 4, 6],
 [11,13,15],
 [12,14,16]]

both row-col and col-row binary search for 13 will fail!
```


奇妙遍历法，从右上角开始。

```cpp
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int H = matrix.size(), W = matrix[0].size();
        // not 2d binary search... there is no such thing!
        int i = 0, j = W - 1;
        while (i < H && j >= 0) {
            //cout << i <<","<< j << " = " << matrix[i][j] << endl;
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] < target) i++;
            else j--;
        }
        return false;
    }
};
```

