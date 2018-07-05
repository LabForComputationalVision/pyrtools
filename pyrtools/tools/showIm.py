import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# from PIL import ImageTk
# import PIL
# import scipy.stats
# import tkinter
# import math
# from .utils import matlab_round


import wx
def make_figure(n_rows, n_cols, axis_size_pix, col_margin_pix=10, row_margin_pix=10):
    app = wx.App(False)
    ppi_x, ppi_y = wx.ScreenDC().GetPPI()
    assert ppi_x == ppi_y, "ppi must be same in both directions!"
    ppi = ppi_x
    # add extra 20% to the y direction for extra info
    fig = plt.figure(figsize=(((n_cols-1)*col_margin_pix+n_cols*axis_size_pix[1]) / ppi, ((n_rows-1)*row_margin_pix+n_rows*(axis_size_pix[0]/.8)) / ppi), dpi=ppi)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*fig.dpi, bbox.height*fig.dpi
    rel_axis_width = axis_size_pix[1] / fig_width
    rel_axis_height = axis_size_pix[0] / fig_height
    rel_col_margin = col_margin_pix / fig_width
    rel_row_margin = row_margin_pix / fig_height
    for i in range(n_rows):
        for j in range(n_cols):
            fig.add_axes([j*(rel_axis_width+rel_col_margin), 1.-((i+1)*rel_axis_height/.8+i*rel_row_margin), rel_axis_width, rel_axis_height], frameon=False, xticks=[],yticks=[])
    return fig


def _showIm(img, ax, vrange, zoom, title='', cmap=cm.gray, **kwargs):
    ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1], interpolation='none', **kwargs)
    # 12 pt font looks good on axes that 256 pixels high, so we stick with that ratio
    ax.set_title(title + '\n range: [{:0.1f},{:0.1f}] \n dims: [{},{}] * {}'.format(
                 vrange[0], vrange[1], img.shape[0], img.shape[1], zoom), {'fontsize': ax.bbox.height*(12./256)})



def reshape_axis(ax, axis_size_pix):
    if ax.bbox.width < axis_size_pix[1] or ax.bbox.height < axis_size_pix[0]:
        raise Exception("Your axis is too small! Axis size: ({}, {}). Image size: ({}, {})".format(ax.bbox.width, ax.bbox.height, axis_size_pix[1], axis_size_pix[0]))
    bbox = ax.figure.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*ax.figure.dpi, bbox.height*ax.figure.dpi
    rel_axis_width = axis_size_pix[1] / fig_width
    rel_axis_height = axis_size_pix[0] / fig_height
    ax.set_position([*ax.get_position().bounds[:2], rel_axis_width, rel_axis_height])
    return ax


def showIm(img, vrange=None, zoom=1, title='', nshades=256, ax=None, cmap=cm.gray, col_wrap=None, **kwargs):
    '''temporary fix

    img: 2d array (one image to display), 3d array (multiple images to display, images are indexed along the first dimension), or list of 2d arrays.

    title: string or list of strings. if string, will put the same title on every plot. if list of strings, must be the same length as img, and will
           assume that titles go with the corresponding image

    col_wrap:

    ax: matplotlib axis or None. If None, make the appropriate figure. If not None, we reshape it (which we only do by shrinking the bbox, so if the bbox is already too small, this will throw an Exception!)
        so that it's the appropriate number of pixels.

    TODO:
    range, nshades
    '''

    img = np.array(img)
    if img.ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
    if ax is None:
        if col_wrap is None:
            n_cols = img.shape[0]
            n_rows = 1
        else:
            n_cols = col_wrap
            n_rows = int(np.ceil(img.shape[0] / n_cols))
        fig = make_figure(n_rows, n_cols, zoom*np.array(img.shape[1:]))
        axes = fig.axes
    else:
        fig = ax.figure
        axes = [reshape_axis(ax, img.shape[1:])]
    if not isinstance(title, list):
        title = len(img) * [title]
    else:
        assert len(img) == len(title), "Must have same number of titles and images!"
    if vrange is None:
        vrange = [np.min(img), np.max(img)]
    for ax, im, t in zip(axes, img, title):
        _showIm(im, ax, vrange, zoom, t, cmap, **kwargs)
    return fig


# def showIm( img=[], range=None, zoom=[1], label='', colormap=plt.cm.gray, ncols=None):
#     '''under development
#
#     TODO:
#     range, zoom, label, nshades, ncols
#
#     return axes, not fig
#
#     '''
#
#     if range is None:
#         range = [np.min(img), np.max(img)]
#     elif range == 'percentile':
#         # TODO
#         range = [np.min(img), np.max(img)]
#
#     # TODO
#     ncols = len(img)
#
#     fig, axes = plt.subplots(ncols=ncols)
#
#     for i, im in enumerate(img):
# #         row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col)
# #         cax = axes[row,col]
#         ax = axes[i]
#         im = np.array(im).astype(float)
#         ax.imshow(im, cmap=colormap, interpolation='none', vmin=range[0], vmax=range[1])
#         ax.axis('off')
#     # plt.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, data=None)
#     # TODO
#     fig.suptitle(label + '\n range: [{:0.1f},{:0.1f}] \n dims: [{},{}]'.format(
#               np.min(img), np.max(img), im.shape[0], im.shape[1])
#               )
#     fig.tight_layout()
# #     plt.subplots_adjust(top=0.9)
#     plt.show()


# from PIL import ImageTk
# import PIL
# import scipy.stats
# import tkinter
# import math
# from .utils import matlab_round


# def showIm(*args):
#     # check and set input parameters
#     if len(args) == 0:
#         print("showIm( matrix, range, zoom, label, nshades )")
#         print("  matrix is string. It should be the name of a 2D array.")
#         print("  range is a two element tuple.  It specifies the values that ")
#         print("    map to the min and max colormap values.  Passing a value ")
#         print("    of 'auto' (default) sets range=[min,max].  'auto2' sets ")
#         print("    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets ")
#         print("    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th ")
#         print("    percientile value of the sorted matix samples, and p2 is ")
#         print("    the 90th percentile value.")
#         print("  zoom specifies the number of matrix samples per screen pixel.")
#         print("    It will be rounded to an integer, or 1 divided by an ")
#         print("    integer.")
#         #print "    A value of 'same' or 'auto' (default) causes the "
#         #print "    zoom value to be chosen automatically to fit the image into"
#         #print "    the current axes."
#         #print "    A value of 'full' fills the axis region "
#         #print "    (leaving no room for labels)."
#         print("  label - A string that is used as a figure title.")
#         print("  NSHADES (optional) specifies the number of gray shades, ")
#         print("    and defaults to the size of the current colormap. ")
#
#     if len(args) > 0:   # matrix entered
#         matrix = np.array(args[0])
#
#     if len(args) > 1:   # range entered
#         if isinstance(args[1], str):
#             if args[1] is "auto":
#                 imRange = ( np.amin(matrix), np.amax(matrix) )
#             elif args[1] is "auto2":
#                 imRange = ( matrix.mean()-2*matrix.std(),
#                             matrix.mean()+2*matrix.std() )
#             elif args[1] is "auto3":
#                 #p1 = np.percentile(matrix, 10)  not in python 2.6.6?!
#                 #p2 = np.percentile(matrix, 90)
#                 p1 = scipy.stats.scoreatpercentile(np.hstack(matrix), 10)
#                 p2 = scipy.stats.scoreatpercentile(np.hstack(matrix), 90)
#                 imRange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)
#             else:
#                 print("Error: range of %s is not recognized." % args[1])
#                 print("       please use a two element tuple or ")
#                 print("       'auto', 'auto2' or 'auto3'")
#                 print("       enter 'showIm' for more info about options")
#                 return
#         else:
#             imRange = args[1][0], args[1][1]
#     else:
#         imRange = ( np.amin(matrix), np.amax(matrix) )
#
#     if len(args) > 2:   # zoom entered
#         zoom = args[2]
#     else:
#         zoom = 1
#
#     if len(args) > 3:   # label entered
#         label = args[3]
#     else:
#         label = ''
#
#     if len(args) > 4:   # colormap entered
#         nshades = args[4]
#     else:
#         nshades = 256
#
#     # create window
#     #master = Tkinter.Tk()
#     master = tkinter.Toplevel()
#     master.title('showIm')
#     canvas_width = matrix.shape[0] * zoom
#     canvas_height = matrix.shape[1] * zoom
#     master.geometry(str(canvas_width+20) + "x" + str(canvas_height+60) +
#                     "+200+200")
#     # put in top spacer
#     spacer = tkinter.Label(master, text='').pack()
#
#     # create canvas
#     canvas = tkinter.Canvas(master, width=canvas_width, height=canvas_height)
#     canvas.pack()
#     # shift matrix to 0.0-1.0 then to 0-255
#     if (matrix < 0).any():
#         matrix = matrix + math.fabs(matrix.min())
#     matrix = (matrix / matrix.max()) * 255.0
#     print(matrix.astype('uint8')[0,:])
#     img = PIL.Image.fromarray(matrix.astype('uint8'))
#
#     # make colormap
#     colorTable = [0] * 256
#     incr = int(np.ceil(float(matrix.max()-matrix.min()+1) / float(nshades)))
#     colors = list(range(int(matrix.min()), int(matrix.max())+1, incr))
#     colors[0] = 0
#     colors[-1] = 255
#     colctr = -1
#     # compute color transition indices
#     thresh = matlab_round( (matrix.max() - matrix.min()) / len(colors) )
#     for i in range(len(colorTable)):
#         # handle uneven color boundaries
#         if thresh == 0 or (i % thresh == 0 and colctr < len(colors)-1):
#             colctr += 1
#         colorTable[i] = colors[colctr]
#     img = img.point(colorTable)
#
#     # zoom
#     if zoom != 1:
#         img = img.resize((canvas_width, canvas_height), Image.NEAREST)
#
#     # apply image to canvas
#     imgPI = ImageTk.PhotoImage(img)
#     canvas.create_image(0,0, anchor=tkinter.NW, image=imgPI)
#
#     # add labels
#     rangeStr = 'Range: [%.1f, %.1f]' % (imRange[0], imRange[1])
#     rangeLabel = tkinter.Label(master, text=rangeStr).pack()
#     dimsStr = 'Dims: [%d, %d] / %d' % (matrix.shape[0], matrix.shape[1], zoom)
#     dimsLabel = tkinter.Label(master, text=dimsStr).pack()
#
#     tkinter.mainloop()
