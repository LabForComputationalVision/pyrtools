def strictly_decreasing(L):
    ''' are all elements of list strictly decreasing '''
    return all(x>y for x, y in zip(L, L[1:]))
