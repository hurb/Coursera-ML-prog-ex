__author__ = 'Sony'
import sys
from numpy import *
def f(x, y):
    x = x**2
    y = y**3
    out = hstack(x)
    out = hstack((out, y))
    print(shape(out))
    return out

def main():
    print(f(array([1, 2, 3]), array([1, 2, 3])))


if __name__ == '__main__':
    main()