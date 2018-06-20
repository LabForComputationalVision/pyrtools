from .binomialFilter import binomialFilter
import math
import numpy

def namedFilter(name):
    ''' Some standard 1D filter kernels. These are scaled such that their 
        L2-norm is 1.0

        binomN              - binomial coefficient filter of order N-1
        haar                - Harr wavelet
        qmf8, qmf12, qmf16  - Symmetric Quadrature Mirror Filters [Johnston80]
        daub2, daub3, daub4 - Daubechies wavelet [Daubechies88]
        qmf5, qmf9, qmf13   - Symmetric Quadrature Mirror Filters 
                              [Simoncelli88, Simoncelli90]
        [Johnston80] - J D Johnston, "A filter family designed for use in 
                       quadrature mirror filter banks", Proc. ICASSP, 
                       pp 291-294, 1980.
        [Daubechies88] - I Daubechies, "Orthonormal bases of compactly 
                         supported wavelets", Commun. Pure Appl. Math, vol. 42,
                         pp 909-996, 1988.
        [Simoncelli88] - E P Simoncelli,  "Orthogonal sub-band image 
                         transforms", PhD Thesis, MIT Dept. of Elec. Eng. and 
                         Comp. Sci. May 1988. Also available as: MIT Media 
                         Laboratory Vision and Modeling Technical Report #100.
        [Simoncelli90] -  E P Simoncelli and E H Adelson, "Subband image 
                          coding", Subband Transforms, chapter 4, ed. John W 
                          Woods, Kluwer Academic Publishers,  Norwell, MA, 1990,
                          pp 143--192.   '''

    if len(name) > 5 and name[:5] == "binom":
        kernel = math.sqrt(2) * binomialFilter(int(name[5:]))
    elif name is "qmf5":
        kernel = numpy.array([[-0.076103], [0.3535534], [0.8593118], [0.3535534], [-0.076103]])
    elif name is "qmf9":
        kernel = numpy.array([[0.02807382], [-0.060944743], [-0.073386624], [0.41472545], [0.7973934], [0.41472545], [-0.073386624], [-0.060944743], [0.02807382]])
    elif name is "qmf13":
        kernel = numpy.array([[-0.014556438], [0.021651438], [0.039045125], [-0.09800052], [-0.057827797], [0.42995453], [0.7737113], [0.42995453], [-0.057827797], [-0.09800052], [0.039045125], [0.021651438], [-0.014556438]])
    elif name is "qmf8":
        kernel = math.sqrt(2) * numpy.array([[0.00938715], [-0.07065183], [0.06942827], [0.4899808], [0.4899808], [0.06942827], [-0.07065183], [0.00938715]])
    elif name is "qmf12":
        kernel = math.sqrt(2) * numpy.array([[-0.003809699], [0.01885659], [-0.002710326], [-0.08469594], [0.08846992], [0.4843894], [0.4843894], [0.08846992], [-0.08469594], [-0.002710326], [0.01885659], [-0.003809699]])
    elif name is "qmf16":
        kernel = math.sqrt(2) * numpy.array([[0.001050167], [-0.005054526], [-0.002589756], [0.0276414], [-0.009666376], [-0.09039223], [0.09779817], [0.4810284], [0.4810284], [0.09779817], [-0.09039223], [-0.009666376], [0.0276414], [-0.002589756], [-0.005054526], [0.001050167]])
    elif name is "haar":
        kernel = numpy.array([[1], [1]]) / math.sqrt(2)
    elif name is "daub2":
        kernel = numpy.array([[0.482962913145], [0.836516303738], [0.224143868042], [-0.129409522551]]);
    elif name is "daub3":
        kernel = numpy.array([[0.332670552950], [0.806891509311], [0.459877502118], [-0.135011020010], [-0.085441273882], [0.035226291882]])
    elif name is "daub4":
        kernel = numpy.array([[0.230377813309], [0.714846570553], [0.630880767930], [-0.027983769417], [-0.187034811719], [0.030841381836], [0.032883011667], [-0.010597401785]])
    elif name is "gauss5":  # for backward-compatibility
        kernel = math.sqrt(2) * numpy.array([[0.0625], [0.25], [0.375], [0.25], [0.0625]])
    elif name is "gauss3":  # for backward-compatibility
        kernel = math.sqrt(2) * numpy.array([[0.25], [0.5], [0.25]])
    else:
        print("Error: Bad filter name: %s" % (name))
        exit(1)

    return numpy.array(kernel)
