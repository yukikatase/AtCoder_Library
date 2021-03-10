import bisect, copy, heapq, math, sys
from collections import *
from functools import lru_cache
from itertools import accumulate, combinations, permutations, product
def input():
    return sys.stdin.readline()[:-1]
def ruiseki(lst):
    return [0]+list(accumulate(lst))
def ceil(a,b):
    return -(-a//b)
def create_graph(N,edge):
    g=[[] for i in range(N)]
    for i,j in edge:
        i,j=i-1,j-1
        g[i].append(j)
        g[j].append(i)
    return g
sys.setrecursionlimit(5000000)
mod=pow(10,9)+7
INF = 1 << 30
al=[chr(ord('a') + i) for i in range(26)]
direction=[[1,0],[0,1],[-1,0],[0,-1]]

