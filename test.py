# 高速な入力
import sys
def input():
    return sys.stdin.readline()[:-1]


# 桁数を指定して出力
pi = 3.141592
'{:.1f}'.format(pi)
# 3.1
'{:.2f}'.format(pi)

s="1234"
s.zfill(8)
# 00001234
s="-1234"
s.zfill(8)
# -0001234


# 累積和
from itertools import accumulate
a = list(range(1, 11))
b = list(accumulate(a))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]


# 順列
from itertools import permutations
list(permutations([1, 2, 3]))
# [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


# 二分探索
import bisect
A = [1, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6]
bisect.bisect_left(A, 3)
# 2 最も左(前)の挿入箇所が返ってきている
index = bisect.bisect_right(A, 3)
# 5


# 辞書
dict1 = {'a': 1, 'b': 2, 'c': 3}
list(dict1.keys())
# ['a', 'b', 'c']
list(dict1.values())
# [1, 2, 3]
list(dict1.items())
# [('a', 1), ('b', 2), ('c', 3)]


# math.factorial(n) でnの階乗
# 大きすぎたり何度もやるとTLEの可能性あり
import math


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


# 双方向リスト
from collections import deque
d=deque()
d.append(1)
d.append(2)
d.appendleft(3)
# deque([3, 1, 2])
d.pop()
# 2
# deque([3, 1])
d.popleft()
# 2
# deque([1])


from fractions import gcd
# 最大公約数
a = [1, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6]
ans = a[0]
for i in range(1, N):
    ans = gcd(ans, a[i])

# 最小公倍数
ans = a[0]
for i in range(1, N):
    ans = ans * a[i] // gcd(ans, a[i])


# 文字をアスキーコードへ
ord(a)
# アスキーコードを文字へ
chr(97)
al=[chr(ord('a') + i) for i in range(26)]
print(al)
# ['a', 'b', 'c', 'd', 'e',~~, 'z']


# 安いcombination
# TLEの可能性めちゃめちゃあり
def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
# MOD combination
def cmb(n, r, mod=10**9+7):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

mod = 10**9+7 #出力の制限
N = 10**4
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, N + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )


# 素数判定
def is_prime(n):
    if n == 1: return False

    for k in range(2, int(math.sqrt(n)) + 1):
        if n % k == 0:
            return False

    return True


# 素因数分解
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


# Union Find
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

    def union(self, x, y, w):
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


# ワーシャルフロイド
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