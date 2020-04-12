import sys
from collections import *
import heapq
import math
from math import gcd
import bisect
import copy
from itertools import permutations,accumulate,combinations,product
def input():
    return sys.stdin.readline()[:-1]
def ruiseki(lst):
    return [0]+list(accumulate(lst))
mod=pow(10,9)+7
al=[chr(ord('a') + i) for i in range(26)]
direction=[[1,0],[0,1],[-1,0],[0,-1]]

