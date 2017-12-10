def nextSz(size, sizeList):
    ''' find next largest size in pyramid list '''

    ## make sure sizeList is strictly increasing
    if sizeList[0] > sizeList[len(sizeList)-1]:
        sizeList = sizeList[::-1]
    outSize = (0,0)
    idx = 0;
    while outSize == (0,0) and idx < len(sizeList):
        if sizeList[idx] > size:
            outSize = sizeList[idx]
        idx += 1

    return outSize
