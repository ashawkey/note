# 排序 (Sorting)

## 基本概念

- **稳定/不稳定**：相同关键码的记录在排序后保持原有相对顺序的为稳定排序。
- **空间代价**：指**额外**空间。
- **时间代价**：记 Best ~ Average ~ Worst。
- **下界**（判定树）：n 个记录有 n! 种排列，判定树深度 $O(\lg n!) \sim O(n\log n)$——基于比较的排序**最坏不可能快于 $O(n\log n)$**。

## 简单排序

| 算法 | 稳定 | 时间 (B ~ A ~ W) | 空间 | 要点 |
| --- | --- | --- | --- | --- |
| 插入排序 | ✅ | $O(n) \sim O(n^2) \sim O(n^2)$ | $O(1)$ | 序列基本有序时快；二分优化比较到 $O(n\log n)$ 但移动仍 $O(n^2)$ |
| 冒泡排序 | ✅ | $O(n) \sim O(n^2) \sim O(n^2)$ | $O(1)$ | 一轮无交换可提前结束 |
| 选择排序 | ❌ | $O(n^2) \sim O(n^2) \sim O(n^2)$ | $O(1)$ | 交换破坏稳定性；改进方向是堆排序 |
| 希尔排序 | ❌ | 取决于增量序列（Hibbard 序列 $O(n^{3/2})$） | $O(1)$ | 缩小增量插入排序 |

```c++
// 插入排序（移动代替交换）
void insertsort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int tmp = arr[i];
        int j = i - 1;
        while (j >= 0 && tmp < arr[j]) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = tmp;
    }
}
```

## 快排 (Quick Sort)

**不稳定**。$O(n\log n) \sim O(n\log n) \sim O(n^2)$（已有序时退化）。平均复杂度分析类似随机 BST：

$$
T(n) = \frac 2 n \sum_{i=0}^{n-1} T(i) + cn \Rightarrow T(n) \sim O(n\log n)
$$

```c++
int partition(int arr[], int left, int right) {
    int i = left, j = right;
    int tmp = arr[left]; // pivot is selected as the left.
    while (i != j) {
        while ((arr[j] > tmp) && (i < j)) j--; // right to left
        if (i < j) arr[i++] = arr[j];
        while ((arr[i] <= tmp) && (i < j)) i++; // left to right
        if (i < j) arr[j--] = arr[i];
    }
    arr[i] = tmp;
    return i;
}

void quicksort(int arr[], int left, int right) {
    if (left < right) {
        int pivot = partition(arr, left, right);
        quicksort(arr, left, pivot - 1);
        quicksort(arr, pivot + 1, right);
    }
}
```

### 变体：第 k 小 / 最小的前 k 个

**第 k 小（不要求前 k 个有序）**：快排分治只搜一侧，平均 $O(n)$：

```c++
void quicksort_k(int arr[], int l, int r, int k) {
    if (l < r) {
        int i = l, j = r, x = arr[l];
        while (i < j) {
            while (i < j && arr[j] >= x) j--;
            if (i < j) arr[i++] = arr[j];
            while (i < j && arr[i] <= x) i++;
            if (i < j) arr[j--] = arr[i];
        }
        arr[i] = x;
        int tmp = i - l + 1; // 小于等于 pivot 的个数
        if (k == tmp) return;
        else if (k < tmp) quicksort_k(arr, l, i - 1, k);
        else quicksort_k(arr, i + 1, r, k - tmp);
    }
}
// std::nth_element(begin, kth, end) 实现了这一点
```

**有序的前 k 个最小元素**（复杂度都含 k）：

- 完整快排 $O(n\log n)$；小根堆 $O(n + k\log n)$；
- **容量为 k 的大根堆** $O(n\log k)$：`std::partial_sort(arr, arr+k, arr+N)`：

```c++
priority_queue<int> heap_k(int arr[], int n, int k) {
    priority_queue<int> q; // maxheap
    for (int i = 0; i < k; i++) q.push(arr[i]);
    for (int i = k; i < n; i++) {
        if (arr[i] < q.top()) {
            q.pop();
            q.push(arr[i]);
        }
    }
    return q;
}
```

## 归并排序 (Merge Sort)

**稳定**，时间稳定 $O(n\log n)$，空间 $O(n)$。快排关注 divide，归并关注 merge。

```c++
void mergesort(int arr[], int tmp[], int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        mergesort(arr, tmp, left, mid);
        mergesort(arr, tmp, mid + 1, right);
        merge(arr, tmp, left, right, mid);
    }
}

void merge(int arr[], int tmp[], int left, int right, int mid) {
    for (int i = left; i <= right; i++) tmp[i] = arr[i];
    int i = left, j = mid + 1, idx = left;
    while (i <= mid && j <= right) {
        if (tmp[i] <= tmp[j]) arr[idx++] = tmp[i++];
        else arr[idx++] = tmp[j++];
    }
    while (i <= mid) arr[idx++] = tmp[i++];
    while (j <= right) arr[idx++] = tmp[j++];
}
```

**变体应用**：归并过程中统计**逆序对**（左边大于右边时 `ans += m - i + 1`），见 `52_complexity`。

## 堆排序

选择排序的高级版，**不稳定**，$O(n\log n)$ 稳定，$O(1)$ 空间（原地）：

```c++
void heapsort(int arr[], int n) {
    priority_queue<int> que(arr, arr + n);
    for (int i = 0; i < n; i++) arr[i] = que.top(), que.pop(); // greater to less
}
```

堆的 siftdown/siftup/建堆实现见 `12_heap`。

## 桶排序 / 基数排序（非比较排序）

不通过比较与交换，而通过**收集与分配**。需要值域假设。

**计数排序**（值域小 [0, m) 时适用）：稳定（倒序收集），$O(m+n)$：

```c++
void bucketsort(int arr[], int n, int m) {  // assume arr[]'s range in [0, m)
    int* tmp = new int[n];
    int* cnt = new int[m];
    for (int i = 0; i < n; i++) tmp[i] = arr[i];
    for (int i = 0; i < m; i++) cnt[i] = 0;
    for (int i = 0; i < n; i++) cnt[arr[i]]++;
    for (int i = 1; i < m; i++) cnt[i] += cnt[i - 1];
    // collect in reverse order to keep stability.
    for (int i = n - 1; i >= 0; i--) arr[--cnt[tmp[i]]] = tmp[i];
}
```

**基数排序**（值域大时）：LSD 从低位到高位做 d 次计数排序。稳定，$O(d \cdot (r + n))$，但 $d \ge \log_r n$，故总体 $\sim O(n\log n)$：

```c++
void radixsort(int arr[], int n, int d, int r) {  // d 位数，每位 [0, r)
    int* tmp = new int[n];
    int* cnt = new int[r];
    int radix = 1;
    for (int i = 1; i <= d; i++) {  // LSD
        for (int j = 0; j < r; j++) cnt[j] = 0;
        for (int j = 0; j < n; j++) cnt[(arr[j] / radix) % r]++;
        for (int j = 1; j < r; j++) cnt[j] += cnt[j - 1];
        for (int j = n - 1; j >= 0; j--) {
            int k = (arr[j] / radix) % r;
            tmp[--cnt[k]] = arr[j];
        }
        for (int j = 0; j < n; j++) arr[j] = tmp[j];
        radix *= r;
    }
}
```

**字符串基数排序**（从首字母开始递归分桶）：字母串排序的性质——从第一个字母开始排序，后面的桶排序不会改变前面的次序。$O(\sum s_i)$。

## 索引排序

对**下标**排序而不是移动记录本身（记录移动代价高时），`adjust` 可做到 $O(1)$ 额外空间地原位调整。

## 外排序

内存放不下时的排序：**置换选择**产生尽可能长的顺串 + **k 路归并**。顺串平均长度 2M（扫雪机模型）；归并用**败者树**把每次取最小从 $O(k)$ 降到 $O(\log k)$。详见 `DA09` 旧笔记。
