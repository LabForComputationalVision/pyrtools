import numpy
from PIL import ImageTk
import PIL
import scipy.stats
import Tkinter
import math
from round import round

def showIm(*args):
    # check and set input parameters
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, nshades )"
        print "  matrix is string. It should be the name of a 2D array."
        print "  range is a two element tuple.  It specifies the values that "
        print "    map to the min and max colormap values.  Passing a value "
        print "    of 'auto' (default) sets range=[min,max].  'auto2' sets "
        print "    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets "
        print "    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th "
        print "    percientile value of the sorted matix samples, and p2 is "
        print "    the 90th percentile value."
        print "  zoom specifies the number of matrix samples per screen pixel."
        print "    It will be rounded to an integer, or 1 divided by an "
        print "    integer."
        #print "    A value of 'same' or 'auto' (default) causes the "
        #print "    zoom value to be chosen automatically to fit the image into"
        #print "    the current axes."
        #print "    A value of 'full' fills the axis region "
        #print "    (leaving no room for labels)."
        print "  label - A string that is used as a figure title."
        print "  NSHADES (optional) specifies the number of gray shades, "
        print "    and defaults to the size of the current colormap. "

    if len(args) > 0:   # matrix entered
        matrix = numpy.array(args[0])

    if len(args) > 1:   # range entered
        if isinstance(args[1], basestring):
            if args[1] is "auto":
                imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
            elif args[1] is "auto2":
                imRange = ( matrix.mean()-2*matrix.std(), 
                            matrix.mean()+2*matrix.std() )
            elif args[1] is "auto3":
                #p1 = numpy.percentile(matrix, 10)  not in python 2.6.6?!
                #p2 = numpy.percentile(matrix, 90)
                p1 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 10)
                p2 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 90)
                imRange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)
            else:
                print "Error: range of %s is not recognized." % args[1]
                print "       please use a two element tuple or "
                print "       'auto', 'auto2' or 'auto3'"
                print "       enter 'showIm' for more info about options"
                return
        else:
            imRange = args[1][0], args[1][1]
    else:
        imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
    
    if len(args) > 2:   # zoom entered
        zoom = args[2]
    else:
        zoom = 1

    if len(args) > 3:   # label entered
        label = args[3]
    else:
        label = ''

    if len(args) > 4:   # colormap entered
        nshades = args[4]
    else:
        nshades = 256

    # create window
    #master = Tkinter.Tk()
    master = Tkinter.Toplevel()
    master.title('showIm')
    canvas_width = matrix.shape[0] * zoom
    canvas_height = matrix.shape[1] * zoom
    master.geometry(str(canvas_width+20) + "x" + str(canvas_height+60) +
                    "+200+200")
    # put in top spacer
    spacer = Tkinter.Label(master, text='').pack()
    
    # create canvas
    canvas = Tkinter.Canvas(master, width=canvas_width, height=canvas_height)
    canvas.pack()
    # shift matrix to 0.0-1.0 then to 0-255
    if (matrix < 0).any():
        matrix = matrix + math.fabs(matrix.min())
    matrix = (matrix / matrix.max()) * 255.0
    print matrix.astype('uint8')[0,:]
    img = PIL.Image.fromarray(matrix.astype('uint8'))

    # make colormap
    colorTable = [0] * 256
    incr = int(numpy.ceil(float(matrix.max()-matrix.min()+1) / float(nshades)))
    colors = range(int(matrix.min()), int(matrix.max())+1, incr)
    colors[0] = 0
    colors[-1] = 255
    colctr = -1
    # compute color transition indices
    thresh = round( (matrix.max() - matrix.min()) / len(colors) )
    for i in range(len(colorTable)):
        # handle uneven color boundaries
        if thresh == 0 or (i % thresh == 0 and colctr < len(colors)-1):
            colctr += 1
        colorTable[i] = colors[colctr]
    img = img.point(colorTable)

    # zoom
    if zoom != 1:
        img = img.resize((canvas_width, canvas_height), Image.NEAREST)

    # apply image to canvas
    imgPI = ImageTk.PhotoImage(img)    
    canvas.create_image(0,0, anchor=Tkinter.NW, image=imgPI)

    # add labels
    rangeStr = 'Range: [%.1f, %.1f]' % (imRange[0], imRange[1])
    rangeLabel = Tkinter.Label(master, text=rangeStr).pack()
    dimsStr = 'Dims: [%d, %d] / %d' % (matrix.shape[0], matrix.shape[1], zoom)
    dimsLabel = Tkinter.Label(master, text=dimsStr).pack()
    
    Tkinter.mainloop()
