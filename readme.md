# テンプレ

```python
import sys
from collections import *
import heapq
import math
import bisect
from itertools import permutations,accumulate,combinations,product
from fractions import gcd
def input():
    return sys.stdin.readline()[:-1]
def ruiseki(lst):
    return [0]+list(accumulate(lst))
mod=pow(10,9)+7
al=[chr(ord('a') + i) for i in range(26)]

```

# Tips

## 桁数を指定して出力

```python
pi = 3.141592
'{:.1f}'.format(pi)
# 3.1
'{:.2f}'.format(pi)
# 3.12
```

```python
s="1234"
s.zfill(8)
# 00001234
s="-1234"
s.zfill(8)
# -0001234
```

## アスキーコード関連

```python
# 文字をアスキーコードへ
ord(a)
# アスキーコードを文字へ
chr(97)
al=[chr(ord('a') + i) for i in range(26)]
# ['a', 'b', 'c', 'd', 'e',~~, 'z']
```

## bit演算

論理積: &

論理和: |

XOR: ^

反転: ~

シフト: << or >>

# アルゴリズム集

## 累積和

List Aの累積和が欲しい場合

```python
lst＝ruiseki(A)
```

## 順列

数字を一列に並べた場合の数が得られるやべーやつ

```python
from itertools import permutations
A=[1, 2, 3]
list(permutations(A))
# [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
```

## 二分探索

```python
import bisect
A = [1, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6]
bisect.bisect_left(A, 3)
# 2 最も左(前)の挿入箇所が返ってきている
index = bisect.bisect_right(A, 3)
# 5
```

## 辞書

```python
dict1 = {'a': 1, 'b': 2, 'c': 3}
list(dict1.keys())
# ['a', 'b', 'c']
list(dict1.values())
# [1, 2, 3]
list(dict1.items())
# [('a', 1), ('b', 2), ('c', 3)]
```

## ヒープソート

```python
# ヒープソート関係
import heapq
# リストAをヒープ化(これを初めにしないとダメ)
heapq.heapify(A)
# aをヒープに入れる
heapq.heappush(A, a)
# 最小要素をAから出す
heapq.heappop(A)
# aをヒープに入れてソートされ、最小要素をpop
heapq.heappushpop(A, a)
# 最小要素をpopしaをヒープへ
heapq.heapreplace(A, a)
```

## 最小公倍数

```python
from fractions import gcd
# 最小公倍数
l = 1
def lcm(l, num):
    return l * num // gcd(l, num)
# 例
l=1
for i in range(n):
    l=lcm(l, a[i])
```

## Mod combination

一度for文で作ってから、cmbすれば答えが出てくる

```python
# MOD combination
def cmb(n, r, mod=10**9+7):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

N = 10**5
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, N + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )
```

## 素数判定

```python
def is_prime(n):
    if n == 1: return False
    for k in range(2, int(math.sqrt(n)) + 1):
        if n % k == 0:
            return False
    return True
```

## 素因数分解

```python
def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])
    if temp!=1:
        arr.append([temp, 1])
    if arr==[]:
        arr.append([n, 1])
    return arr
```

## Union find

```python
class WeightedUnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.weight = [0] * (n+1)


    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            y = self.find(self.par[x])
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    def union(self, x, y, w=1):
        rx = self.find(x)
        ry = self.find(y)
        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
            self.weight[rx] = w - self.weight[x] + self.weight[y]
        else:
            self.par[ry] = rx
            self.weight[ry] = -w - self.weight[y] + self.weight[x]
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def diff(self, x, y):
        return self.weight[x] - self.weight[y]
```

## ワーシャルフロイド

```python
def warshall_floyd(d):
    #d[i][j]: iからjへの最短距離
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j],d[i][k] + d[k][j])
    return d

n,w = map(int,input().split()) #n:頂点数　w:辺の数

d = [[float("inf") for i in range(n)] for i in range(n)] 
#d[u][v] : 辺uvのコスト(存在しないときはinf)
for i in range(w):
    x,y,z = map(int,input().split())
    d[x][y] = z
    d[y][x] = z
for i in range(n):
    d[i][i] = 0 #自身のところに行くコストは０
print(warshall_floyd(d))
```

## Zアルゴリズム

```python
def Z_algo(S):
    n = len(S)
    LCP = [0]*n
    c = 0#最も末尾側までLCPを求めたインデックス
    for i in range(1, n):
        #i番目からのLCPが以前計算したcからのLCPに含まれている場合
        if i+LCP[i-c] < c+LCP[c]:
            LCP[i] = LCP[i-c]
        else:
            j = max(0, c+LCP[c]-i)
            while i+j < n and S[j] == S[i+j]: j+=1
            LCP[i] = j
            c = i
    LCP[0] = n
    return LCP
```

## SegTree

```python
class SegTree:
    # init_valはseg木にしたいlist、ide_eleは単位元
    # """
    # 単位元の説明
    # 最小値のセグ木 → 10**9　(最小値の更新に影響しないため)
    # 　　和のセグ木 → 0　(上の単位元の説明を参照)
    # 　　積のセグ木 → 1　(上の単位元の説明を参照)
    # 　　gcdのセグ木 → 0　(gcdを更新しない値は0)
    # """
    def __init__(self, init_val, ide_ele):
        self.ide_ele=ide_ele
        self.num=2**(len(init_val)-1).bit_length()
        self.seg=[self.ide_ele]*2*self.num

        #set_val
        for i in range(len(init_val)):
            self.seg[i+self.num-1]=init_val[i]
        #built
        for i in range(self.num-2,-1,-1) :
            self.seg[i]=self.segfunc(self.seg[2*i+1],self.seg[2*i+2])

    # 書け
    def segfunc(self, x, y):
        return 
    
    def update(self, k, x):
        k += self.num-1
        self.seg[k] = x
        while k:
            k = (k-1)//2
            self.seg[k] = self.segfunc(self.seg[k*2+1],self.seg[k*2+2])

    def query(self, p, q):
        if q<=p:
            return self.ide_ele
        p += self.num-1
        q += self.num-2
        res=self.ide_ele
        while q-p>1:
            if p&1 == 0:
                res = self.segfunc(res,self.seg[p])
            if q&1 == 1:
                res = self.segfunc(res,self.seg[q])
                q -= 1
            p = p//2
            q = (q-1)//2
        if p == q:
            res = self.segfunc(res,self.seg[p])
        else:
            res = self.segfunc(self.segfunc(res,self.seg[p]),self.seg[q])
        return res
```
