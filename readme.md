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
# 最小公倍数
l = 1
def lcm(l, num):
    return l * num // math.gcd(l, num)
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

N = 10**6
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

## 素因数分解(エラトステネス済み)

```python
def era_primes(n):
    is_prime = [-1] * (n + 1)
    is_prime[1]=1
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]!=-1:
            continue
        for j in range(i * 2, n + 1, i):
            if is_prime[j]==-1:
                is_prime[j] = i
    for i in range(n):
        if is_prime[i+1]==-1:
            is_prime[i+1]=i+1
    return is_prime

def factorize(N,prime_list):  # prime_listに素数のリストをぶち込め！
    lst = []
    # N==1のときも
    # if N==1:
    #     lst.append(1)
    while prime_list[N]!=1:
        lst.append(prime_list[N])
        N//=prime_list[N]
    return lst
```

## 約数列挙

```python
def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)

    divisors.sort()
    return divisors
```

## エラトステネスの篩

```python
def primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if not is_prime[i]:
            continue
        for j in range(i * 2, n + 1, i):
            is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]
```

## Union find

```python
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
        self.group_num = n
 
    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
 
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
 
        if x == y:
            return
 
        self.group_num -= 1
        if self.parents[x] > self.parents[y]:
            x, y = y, x
 
        self.parents[x] += self.parents[y]
        self.parents[y] = x
 
    def size(self, x):
        return -self.parents[self.find(x)]
 
    def same(self, x, y):
        return self.find(x) == self.find(y)
 
    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]
 
    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]
 
    def group_count(self):
        return self.group_num
 
    def all_group_members(self):
        self.all_group_member = defaultdict(list)
        for i in range(self.n):
            self.all_group_member[self.find(i)].append(i)
        return self.all_group_member
 
    def __str__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())
```

## ワーシャルフロイド

```python
# O(V**3)
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
warshall_floyd(d)
```

## ダイクストラ

### ふつうの

```python
# O(E+Vlog(V))
def dijkstra(s):
    hq = [(0, s)]
    heapq.heapify(hq) # リストを優先度付きキューに変換
    cost = [float('inf')] * n # 行ったことのないところはinf
    cost[s] = 0 # 開始地点は0
    while hq:
        c, v = heapq.heappop(hq)
        if c > cost[v]: # コストが現在のコストよりも高ければスルー
            continue
        for d, u in e[v]:
            tmp = d + cost[v]
            if tmp < cost[u]:
                cost[u] = tmp
                heapq.heappush(hq, (tmp, u))
    return cost

n,m = map(int,input().split())
e = [[] for _ in range(n)]
for _ in range(m):
    a,b,t = map(int,input().split())
    a,b = a-1, b-1
    e[a].append((t, b))
    e[b].append((t, a))
```

### 経路復元

```python
def dijkstra_back(s,t,n,w,cost):
    #sからtへの最短経路の経路復元
    prev = [s] * n #最短経路の直前の頂点
    d = [float("inf")] * n
    used = [False] * n
    d[s] = 0
    
    while True:
        v = -1
        for i in range(n):
            if (not used[i]) and (v == -1):
               v = i
            elif (not used[i]) and d[i] < d[v]:
                v = i
        if v == -1:
               break
        used[v] = True
        
        for i in range(n):
            if d[i] > d[v] + cost[v][i]: 
                d[i] = d[v] + cost[v][i]
                prev[i] = v
        
    path = [t]
    while prev[t] != s:
        path.append(prev[t])
        prev[t] = prev[prev[t]]
    path.append(s)
    path = path[::-1]
    return path

################################
n,w = map(int,input().split()) #n:頂点数　w:辺の数

cost = [[float("inf") for i in range(n)] for i in range(n)] 
#cost[u][v] : 辺uvのコスト(存在しないときはinf この場合は10**10)
for i in range(w):
    x,y,z = map(int,input().split())
    cost[x][y] = z
    cost[y][x] = z
```

## ベルマンフォード

```python
# O(VE)
#True : 負の経路が存在する
def find_negative_loop(n,w,es):
    #負の経路の検出
    #n:頂点数, w:辺の数, es[i]: [辺の始点,辺の終点,辺のコスト]
    d = [float("inf")] * n
    d[0] = 0 
    #この始点はどこでもよい
    for i in range(n):
        for j in range(w):
            e = es[j]
            if d[e[1]] > d[e[0]] + e[2]:
                d[e[1]] = d[e[0]] + e[2]
                if i == n-1:
                    return True
    return False


#############################
n,w = map(int,input().split()) #n:頂点数　w:辺の数
es = [[] for i in range(2*w)] #es[i]: [辺の始点,辺の終点,辺のコスト]
for i in range(w):
    x,y,z = map(int,input().split())
    es[2*i] = [x,y,z]
    es[2*i+1] = [y,x,z]
w = w*2
print(find_negative_loop(n,w,es))
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
    # funcは行いたい操作を代入
    # 区間の最小値を求めたいならmin
    # 区間の最大値を求めたいならmax
    # """
    def __init__(self, init_val, ide_ele, func):
        self.ide_ele=ide_ele
        self.num=2**(len(init_val)-1).bit_length()
        self.seg=[self.ide_ele]*2*self.num
        self.func=func

        #set_val
        for i in range(len(init_val)):
            self.seg[i+self.num-1]=init_val[i]
        #built
        for i in range(self.num-2,-1,-1) :
            self.seg[i]=self.segfunc(self.seg[2*i+1],self.seg[2*i+2])

    # 書け
    def segfunc(self, x, y):
        return self.func(x,y)
    
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

## 遅延SegTree

```python
# N: 処理する区間の長さ
class seg_lazy:
    def __init__(self, N, ide_ele, func):
        self.LV = (N-1).bit_length()
        self.N0 = 2**self.LV
        self.data = [ide_ele]*(2*self.N0)
        self.lazy = [None]*(2*self.N0)
        self.ide_ele = ide_ele
        self.func = func

    # 伝搬対象の区間を求める
    def gindex(self, l, r):
        L = (l + self.N0) >> 1; R = (r + self.N0) >> 1
        lc = 0 if l & 1 else (L & -L).bit_length()
        rc = 0 if r & 1 else (R & -R).bit_length()
        for i in range(self.LV):
            if rc <= i:
                yield R
            if L < R and lc <= i:
                yield L
            L >>= 1; R >>= 1

    # 遅延伝搬処理
    def propagates(self, *ids):
        for i in reversed(ids):
            v = self.lazy[i-1]
            if v is None:
                continue
            self.lazy[2*i-1] = self.data[2*i-1] = self.lazy[2*i] = self.data[2*i] = v
            self.lazy[i-1] = None

    # 区間[l, r)をxで更新
    def update(self, l, r, x):
        *ids, = self.gindex(l, r)
        self.propagates(*ids)

        L = self.N0 + l; R = self.N0 + r
        while L < R:
            if R & 1:
                R -= 1
                self.lazy[R-1] = self.data[R-1] = x
            if L & 1:
                self.lazy[L-1] = self.data[L-1] = x
                L += 1
            L >>= 1; R >>= 1
        for i in ids:
            self.data[i-1] = self.func(self.data[2*i-1], self.data[2*i])

    # 区間[l, r)内の最小値を求める
    def query(self, l, r):
        self.propagates(*self.gindex(l, r))
        L = self.N0 + l; R = self.N0 + r

        s = self.ide_ele
        while L < R:
            if R & 1:
                R -= 1
                s = self.func(s, self.data[R-1])
            if L & 1:
                s = self.func(s, self.data[L-1])
                L += 1
            L >>= 1; R >>= 1
        return s
```



## Binary Indexed Tree

```python
# Binary Indexed Tree (Fenwick Tree)
class BIT:
    def __init__(self, n):
        self.n = n
        self.data = [0]*(n+1)
        self.el = [0]*(n+1)
    # [1,i]の足し算
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.data[i]
            i -= i & -i
        return s
    def add(self, i, x):
        # assert i > 0
        self.el[i] += x
        while i <= self.n:
            self.data[i] += x
            i += i & -i
    # [i,j]の足し算
    def get(self, i, j=None):
        if j is None:
            return self.el[i]
        return self.sum(j) - self.sum(i-1)
```



## 最小全域木

```python
import heapq
def prim_heap(edge):
    used = [True] * n #True:不使用
    edgelist = []
    for e in edge[0]:
        heapq.heappush(edgelist,e)
    used[0] = False
    res = 0
    while len(edgelist) != 0:
        minedge = heapq.heappop(edgelist)
        if not used[minedge[1]]:
            continue
        v = minedge[1]
        used[v] = False
        for e in edge[v]:
            if used[e[1]]:
                heapq.heappush(edgelist,e)
        res += minedge[0]
    return res

#########################
n,w = map(int,input().split())

edge = [[] for i in range(n)]
#隣接リスト edge[i]:[コスト,行先]
for i in range(w):
    x,y,z = map(int,input().split())
    edge[x].append((z,y))
    edge[y].append((z,x))
prim_heap(edge)
```

## LCA(最小共通祖先)

```python
from collections import *
class lca:
  	## rootは根の頂点で0からのインデックス
    ## childlstは子のリスト
    ## 例：[[1, 2], [3, 4], [5, 6], [], [], [], []]
    def __init__(self,root,childlst):
        self.n=len(childlst)
        self.log_n=(self.n-1).bit_length()
        self.parent=[[-1]*n for i in range(self.log_n)]
        self.depth=[-1]*n

        d=deque([[root,-1,0]])
        while d:
            v,prnt,dpth=d.popleft()
            self.parent[0][v]=prnt
            self.depth[v]=dpth
            for i in range(len(childlst[v])):
                d.append([childlst[v][i],v,dpth+1])
        
        for i in range(self.log_n-1):
            for j in range(self.n):
                if self.parent[i][j]<0:
                    self.parent[i+1][j]=-1
                else:
                    self.parent[i+1][j]=self.parent[i][self.parent[i][j]]
        
    ## uとvは0始めのインデックス
    def find_lca(self,u,v):
        # vが深い方,uが浅い方
        if self.depth[u]>self.depth[v]:
            u,v=v,u
        for i in range(self.log_n):
            if (self.depth[v]-self.depth[u])>>i&1:
                v=self.parent[i][v]

        if u==v:
            return u
        
        for i in range(self.log_n)[::-1]:
            if self.parent[i][u]!=self.parent[i][v]:
                u=self.parent[i][u]
                v=self.parent[i][v]
        
        return self.parent[0][u]
```

## 最長増加部分列(LIS)

```python
import bisect

LIS = [seq[0]]
for i in range(len(seq)):
    if seq[i] > LIS[-1]:
        LIS.append(seq[i])
    else:
        LIS[bisect.bisect_left(LIS, seq[i])] = seq[i]

print(len(LIS))
```

## 謎FFT

```python
n=int(input())
ab=[list(map(int,input().split())) for i in range(n)]

po=1
while 2*n>=po:
    po*=2

a,b=[0]*po,[0]*po
for i in range(n):
    a[i+1]=ab[i][0]
    b[i+1]=ab[i][1]

# print(po)
def fft(f,n,ii=1):
    if n==1:
        return f
    f0=[f[2*i+0] for i in range(n//2)]
    f1=[f[2*i+1] for i in range(n//2)]

    f0=fft(f0,n//2,ii)
    f1=fft(f1,n//2,ii)

    zeta=complex(math.cos(2*math.pi/n), math.sin(2*math.pi/n)*ii)
    pow_zeta=1
    for i in range(n):
        f[i]=f0[i%(n//2)]+pow_zeta*f1[i%(n//2)]
        pow_zeta*=zeta
    return f

def i_fft(f,n):
    f=fft(f,n,-1)
    for i in range(n):
        f[i]=f[i]/po
    return f

a2=fft(a,po)
b2=fft(b,po)

ab=[0]*po
for i in range(po):
    ab[i]=a2[i]*b2[i]

c=i_fft(ab,po)


for i in range(1,1+n*2):
    # print(c[i],c[i].real,round(c[i].real))
    print(round(c[i].real))
```

## トポロジカルソート

```python
# V: 頂点数
# G[v] = [w, ...]:
#    有向グラフ上の頂点vから到達できる頂点w
# deg[v]:
#    頂点vに到達できる頂点の数

from collections import deque
ans = list(v for v in range(V) if deg[v]==0)
deq = deque(ans)
used = [0]*V

while deq:
    v = deq.popleft()
    for t in g[v]:
        deg[t] -= 1
        if deg[t]==0:
            deq.append(t)
            ans.append(t)

# ans: トポロジカル順序に並べた頂点
```

## 中国剰余定理

```python
def inv_gcd(a, b):
    a %= b
    if a == 0: return b, 0
    # 初期状態
    s, t = b, a
    m0, m1 = 0, 1
    while t:
        # 遷移の準備
        u = s // t

        # 遷移
        s -= t * u
        m0 -= m1 * u

        # swap
        s, t = t, s
        m0, m1 = m1, m0

    if m0 < 0: m0 += b // s
    return s, m0

def crt(r, m):
    assert len(r) == len(m)
    n = len(r)
    r0, m0 = 0, 1  # 初期値 x = 0 (mod 1)
    for i in range(n):
        assert m[i] >= 1

        #r1, m1は遷移に使う値
        r1, m1 = r[i] % m[i], m[i]

        #m0がm１以上になるようにする。
        if m0 < m1:
            r0, r1 = r1, r0
            m0, m1 = m1, m0

        # m0がm1の倍数のとき gcdはm1、lcmはm0
        # 解が存在すれば何も変わらないので以降の手順はスキップ
        if m0 % m1 == 0:
            if r0 % m1 != r1: return [0, 0]
            continue

        #  拡張ユークリッドの互除法によりgcd(m0, m1)と m0 * im = gcd (mod m1) を満たす imを求める
        g, im = inv_gcd(m0, m1)

        # 解の存在条件の確認
        if (r1 - r0) % g: return [0, 0]

        """
        r0, m0の遷移
        コメントアウト部分はACLでの実装
        C++なのでlong longを超えないようにしている
        C++ はlcm(m0, m1)で割った余りが負になり得る
        """
        # u1 = m1 // g
        # x = (r1 - r0) // g % u1 * im % u1
        # r0 += x * m0
        u1 = m0 * m1 // g
        r0 += (r1 - r0) // g * m0 * im % u1
        m0 = u1
        #if r0 < 0: r0 += m0

    return [r0, m0]
```

