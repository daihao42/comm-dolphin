#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

class KLDivergence():
    def __init__(self):
        pass

    @staticmethod
    def counter(A, B):
        return sum(a * math.log(a/b) for (a,b) in zip(A,B))

if __name__ == '__main__':
    A = [0.99,0.01]
    B = [0.1,0.9]
    print(KLDivergence.counter(A,B))
