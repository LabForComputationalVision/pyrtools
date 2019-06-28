import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib import animation
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML, animshow must be called with "
                  "as_html5=False")
from ..pyramids import convert_pyr_coeffs_to_pyr


class PyrFigure(Figure):
    """custom figure class to ensure that plots are created and saved with a constant dpi

    NOTE: generally, you shouldn't use this directly, relying instead on the make_figure
    function.

    If you do want to use, do the following: fig = plt.figure(FigureClass=PyrFigure) (NOT fig =
    PyrFigure()) and analogously for other pyplot functions (plt.subplots, etc)

    this enables us to make sure there's no aliasing: a single value in the (image) array that
    we're plotting will be represented as an integer multiple of pixels in the displayed figure

    The dpi that's chosen is an arbitrary value, the only thing that matters is that we use the
    same one when creating and saving the figure, which is what we ensure here. This also means
    that you will be unable to use plt.tight_layout, since we set the spacing and size of the
    subplots very intentionally.
    """
    def __init__(self, dpi=96, *args, **kwargs):
        kwargs['dpi'] = dpi
        Figure.__init__(self, *args, **kwargs)

    def savefig(self, fname, dpi_multiple=1, **kwargs):
        """Save the current figure.

        Call signature::

          savefig(fname, dpi_multiple=1, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format=None,
                  transparent=False, bbox_inches=None, pad_inches=0.1,
                  frameon=None)

        The output formats available depend on the backend being used.

        Arguments
        ----------

        fname : str or file-like object
            A string containing a path to a filename, or a Python
            file-like object, or possibly some backend-dependent object
            such as :class:`~matplotlib.backends.backend_pdf.PdfPages`.

            If *format* is *None* and *fname* is a string, the output
            format is deduced from the extension of the filename. If
            the filename has no extension, the value of the rc parameter
            ``savefig.format`` is used.

            If *fname* is not a string, remember to specify *format* to
            ensure that the correct backend is used.

        Other Arguments
        ----------------

        dpi_multiple : [ scalar integer > 0 ]
            How to scale the figure's dots per inch (must be an integer to
            prevent aliasing). Default is 1, equivalent to using to the value
            of the figure (default matplotlib savefig behavior with dpi='figure')

        facecolor : color spec or None, optional
            the facecolor of the figure; if None, defaults to savefig.facecolor

        edgecolor : color spec or None, optional
            the edgecolor of the figure; if None, defaults to savefig.edgecolor

        orientation : {'landscape', 'portrait'}
            not supported on all backends; currently only on postscript output

        papertype : str
            One of 'letter', 'legal', 'executive', 'ledger', 'a0' through
            'a10', 'b0' through 'b10'. Only supported for postscript
            output.

        format : str
            One of the file extensions supported by the active
            backend.  Most backends support png, pdf, ps, eps and svg.

        transparent : bool
            If *True*, the axes patches will all be transparent; the
            figure patch will also be transparent unless facecolor
            and/or edgecolor are specified via kwargs.
            This is useful, for example, for displaying
            a plot on top of a colored background on a web page.  The
            transparency of these patches will be restored to their
            original values upon exit of this function.

        frameon : bool
            If *True*, the figure patch will be colored, if *False*, the
            figure background will be transparent.  If not provided, the
            rcParam 'savefig.frameon' will be used.

        bbox_inches : str or `~matplotlib.transforms.Bbox`, optional
            Bbox in inches. Only the given portion of the figure is
            saved. If 'tight', try to figure out the tight bbox of
            the figure. If None, use savefig.bbox

        pad_inches : scalar, optional
            Amount of padding around the figure when bbox_inches is
            'tight'. If None, use savefig.pad_inches

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.
        """
        dpi = kwargs.pop('dpi', None)
        if dpi is not None:
            warnings.warn("Ignoring dpi argument: with PyrFigure, we do not use the dpi argument"
                          " for saving, use dpi_multiple instead (this is done to prevent "
                          "aliasing)")
        kwargs['dpi'] = self.dpi * dpi_multiple
        super().savefig(fname, **kwargs)

    def tight_layout(self, renderer=None, pad=1.08, h_pad=None, w_pad=None,
                     rect=None):
        """THIS IS NOT SUPPORTED (we control placement very specifically)
        """
        raise AttributeError("tight_layout is not supported with PyrFigure (we control the "
                             "layout, size, and spacing of these figures quite specifically and "
                             "don't want this kind of automatic changes)")


def make_figure(n_rows, n_cols, axis_size_pix, col_margin_pix=10, row_margin_pix=10, vert_pct=.8):
    """make a nice figure.

    this uses our custom PyrFigure class under the hood.

    Arguments
    ---------
    n_rows : `int`
        the number of rows to create in the figure
    n_cols : `int`
        the number of columns to create in the figure
    axis_size_pix : `tuple`
        2-tuple of ints specifying the size of each axis in the figure in pixels.
    col_margin_pix : `int`
        the number of pixels to leave between each column of axes
    row_margin_pix : `int`
        the number of pixels to leave between each row of axes
    vert_pct : `float`
        must lie between 0 and 1. if less than 1, we leave a little extra room at the top to allow
        a title. for example, if .8, then we add an extra 20% on top to leave room for a title

    Returns
    -------
    fig : `PyrFigure`
        the figure containing the axes at the specified size.
    """
    # this is an arbitrary value
    ppi = 96

    # we typically add extra space to the y direction to leave room for the title. this is
    # controlled by vert_pct: the default value works well if you want a title, and it should be 1
    # if you don't want to use a title
    fig = plt.figure(FigureClass=PyrFigure,
                     figsize=(((n_cols-1)*col_margin_pix+n_cols*axis_size_pix[1]) / ppi,
                              ((n_rows-1)*row_margin_pix+n_rows*(axis_size_pix[0]/vert_pct)) / ppi),
                     dpi=ppi)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*fig.dpi, bbox.height*fig.dpi
    rel_axis_width = axis_size_pix[1] / fig_width
    rel_axis_height = axis_size_pix[0] / fig_height
    rel_col_margin = col_margin_pix / fig_width
    rel_row_margin = row_margin_pix / fig_height
    for i in range(n_rows):
        for j in range(n_cols):
            fig.add_axes([j*(rel_axis_width+rel_col_margin),
                          1.-((i+1)*rel_axis_height/vert_pct+i*rel_row_margin), rel_axis_width,
                          rel_axis_height], frameon=False, xticks=[], yticks=[])
    return fig


def _showIm(img, ax, vrange, zoom, title='', cmap=cm.gray, **kwargs):
    """helper function to display the image on the specified axis

    NOTE: should not be used directly.
    """
    ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1], interpolation='none', **kwargs)

    if title is not None:
        # adapt the precision of displayed range to the order of magnitude of the values
        title = title + '\n range: [{:.1e}, {:.1e}] \n dims: [{}, {}] * {}'
        # 12 pt font looks good on axes that 256 pixels high, so we stick with that ratio
        ax.set_title(title.format(vrange[0], vrange[1], img.shape[0], img.shape[1], zoom),
                     {'fontsize': ax.bbox.height*(12./256)})


def reshape_axis(ax, axis_size_pix):
    """reshape axis to the specified size in pixels

    this will reshape an axis so that the given axis is the specified size in pixels, which we use
    to make sure that an axis is the same size as (or an integer multiple of) the array we're
    trying to display. this is to prevent aliasing

    NOTE: this can only shrink a big axis, not make a small one bigger, and will throw an exception
    if you try to do that.

    Arguments
    ---------
    ax : `matpotlib.pyplot.axis`
        the axis to reshape
    axis_size_pix : `int`
        the target size of the axis, in pixels

    Returns
    -------
    ax : `matplotlib.pyplot.axis`
        the reshaped axis
    """
    if ax.bbox.width < axis_size_pix[1] or ax.bbox.height < axis_size_pix[0]:
        raise Exception("Your axis is too small! Axis size: ({}, {}). Image size: ({}, {})".format(
            ax.bbox.width, ax.bbox.height, axis_size_pix[1], axis_size_pix[0]))
    bbox = ax.figure.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*ax.figure.dpi, bbox.height*ax.figure.dpi
    rel_axis_width = axis_size_pix[1] / fig_width
    rel_axis_height = axis_size_pix[0] / fig_height
    ax.set_position([*ax.get_position().bounds[:2], rel_axis_width, rel_axis_height])
    return ax


def colormap_range(image, vrange='indep1', cmap=None):
    """Find the appropriate ranges for colormaps of provided images

    Arguments
    ---------
    image : `np.array` or `list`
        should be a 2d array (one image to display), 3d array (multiple images to display, images
        are indexed along the first dimension), or list of 2d arrays. the image(s) to be shown.
        all images will be automatically rescaled so they're displayed at the same size. thus,
        their sizes must be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to the minimum and
        maximum value of the colormap, respectively. If a string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute value, and come
                     from the minimum or maximum across all images, whichever has the larger
                    absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the minimum/maximum values
                          across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across all images) minus/
                     plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the 10th/90th percentile
                     values to the 10th/90th percentile of the display intensity range. For
                     example: vmin is the 10th percentile image value minus 1/8 times the
                     difference between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the same absolute value,
                      which comes from either their minimum or maximum value, whichever has the
                      larger absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their minimum/maximum
                      values
        * `'indep2'`: each image has an independent vmin/vmax, which is their mean minus/plus 2
                      std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that the 10th/90th
                      percentile values map to the 10th/90th percentile intensities.

    Returns
    -------
    vrange_list : `list`
        list of tuples, same length as `image`. contains the (vmin, vmax) tuple for each image.

    """
    # this will clip the colormap

    # flatimg is one long 1d array, which enables the min, max, mean, std, and percentile calls to
    # operate on the values from each of the images simultaneously.
    flatimg = np.concatenate([i.flatten() for i in image]).flatten()

    if isinstance(vrange, str):
        if vrange[:4] == 'auto':
            if vrange == 'auto0':
                M = np.nanmax([np.abs(np.nanmin(flatimg)), np.abs(np.nanmax(flatimg))])
                vrange_list = [-M, M]
            elif vrange == 'auto1' or vrange == 'auto':
                vrange_list = [np.nanmin(flatimg), np.nanmax(flatimg)]
            elif vrange == 'auto2':
                vrange_list = [flatimg.nanmean() - 2 * flatimg.nanstd(),
                               flatimg.nanmean() + 2 * flatimg.nanstd()]
            elif vrange == 'auto3':
                p1 = np.nanpercentile(flatimg, 10)
                p2 = np.nanpercentile(flatimg, 90)
                vrange_list = [p1-(p2-p1)/8.0, p2+(p2-p1)/8.0]

            # make sure to return as many ranges as there are images
            vrange_list = [vrange_list] * len(image)

        elif vrange[:5] == 'indep':
            # get independent vrange by calling this function one image at a time
            vrange_list = [colormap_range(im, vrange.replace('indep', 'auto'))[0][0] for im in image]
        else:
            vrange_list, _ = colormap_range(image, vrange='auto1')
            warnings.warn('Unknown vrange argument, using auto1 instead')
    else:
        # in this case, we've been passed two numbers, either as a list or tuple
        if len(vrange) != 2:
            raise Exception("If you're passing numbers to vrange, there must be 2 of them!")
        vrange_list = [tuple(vrange)] * len(image)

    # double check that we're returning the right number of vranges
    assert len(image) == len(vrange_list)

    if cmap is None:
        if '0' in vrange:
            cmap = cm.RdBu
        else:
            cmap = cm.gray

    return vrange_list, cmap


def find_zooms(images):
    """find the zooms necessary to display a list of images

    this convenience function takes a list of images and finds out if they can all be displayed at
    the same size. for this to be the case, there must be an integer for each image such that the
    image can be multiplied by that integer to be the same size as the biggest image.

    Arguments
    ---------
    images : `list`
        list of numpy arrays to check the size of. In practice, these are 1d or 2d, but can in
        principle be any number of dimensions

    Returns
    -------
    zooms : `list`
        list of integers showing how much each image needs to be zoomed
    max_shape : `tuple`
        2-tuple of integers, showing the shape of the largest image in the list
    """
    def check_shape_1d(shapes):
        max_shape = np.max(shapes)
        for s in shapes:
            if not (max_shape % s) == 0:
                raise Exception("All images must be able to be 'zoomed in' to the largest image."
                                "That is, the largest image must be a scalar multiple of all "
                                "images.")
        return max_shape
    # in this case, the two images were different sizes and so numpy can't combine them
    # correctly
    max_shape = []
    for i, _ in enumerate(images[0].shape):
        max_shape.append(check_shape_1d([img.shape[i] for img in images]))
    zooms = []
    for img in images:
        # this checks that there's only one unique value in the list max_shape[i] // img.shape[i],
        # where i indexes through the dimensions; that is, that we zoom each dimension by the same
        # amount. this should then work with an arbitrary number of dimensions (in practice, 1 or
        # 2)
        if len(set([max_shape[i] // img.shape[i] for i in range(img.ndim)])) > 1:
            raise Exception("Both height and width must be multiplied by same amount!")
        zooms.append(max_shape[0] // img.shape[0])
    return zooms, max_shape


def imshow(image, vrange='indep1', zoom=1, title='', col_wrap=None, ax=None,
           cmap=None, plot_complex='rectangular', **kwargs):
    '''show image(s)

    Arguments
    ---------
    image : `np.array` or `list`
        should be a 2d array (one image to display), 3d array (multiple images to display, images
        are indexed along the first dimension), or list of 2d arrays. the image(s) to be shown.
        all images will be automatically rescaled so they're displayed at the same size. thus,
        their sizes must be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to the minimum and
        maximum value of the colormap, respectively. If a string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute value, and come
                     from the minimum or maximum across all images, whichever has the larger
                    absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the minimum/maximum values
                          across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across all images) minus/
                     plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the 10th/90th percentile
                     values to the 10th/90th percentile of the display intensity range. For
                     example: vmin is the 10th percentile image value minus 1/8 times the
                     difference between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the same absolute value,
                      which comes from either their minimum or maximum value, whichever has the
                      larger absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their minimum/maximum
                      values
        * `'indep2'`: each image has an independent vmin/vmax, which is their mean minus/plus 2
                      std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that the 10th/90th
                      percentile values map to the 10th/90th percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer.  If <1, must be 1/d
        where d is a a divisor of the size of the largest image.
    title : `str` , `list` or None
        Title for the plot:

        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img, assigning each
          title to corresponding image.
        * if None, no title will be printed.
    col_wrap : `int` or None
        number of axes to have in each row. If None, will fit all axes in a single row.
    ax : `matplotlib.pyplot.axis` or None
        if None, make the appropriate figure. otherwise, we resize it so that it's the appropriate
        number of pixels (done by shrinking the bbox - if the bbox is already too small, this will
        throw an Exception!, so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.

    Returns
    -------
    fig : `PyrFigure`
        figure containing the plotted images
    '''

    if plot_complex not in ['rectangular', 'polar', 'logpolar']:
        warnings.warn("Don't know how to handle plot_complex value %s, defaulting to "
                      "'rectangular'" % plot_complex)
        plot_complex = 'rectangular'

    try:
        if image.ndim == 2:
            # then this is a single image
            image = [image]
    except AttributeError:
        # then this is a list and we don't do anything
        pass
    if not isinstance(title, list):
        title = len(image) * [title]
    else:
        assert len(image) == len(title), "Must have same number of titles and images!"

    # making sure plotting works for (list of) arrays / torch.tensor
    image_tmp = []
    title_tmp = []
    for img, t in zip(image, title):
        if np.iscomplex(img).any():
            if plot_complex == 'rectangular':
                image_tmp.extend([np.real(img), np.imag(img)])
                title_tmp.extend([t + " real", t + " imaginary"])
            elif plot_complex == 'polar':
                image_tmp.extend([np.abs(img), np.angle(img)])
                title_tmp.extend([t + " amplitude", t + " phase"])
            elif plot_complex == 'logpolar':
                image_tmp.extend([np.log2(np.abs(img)), np.angle(img)])
                title_tmp.extend([t + " log amplitude", t + " phase"])
        else:
            image_tmp.append(np.array(img))
            title_tmp.append(t)
    image = np.array(image_tmp)
    title = title_tmp

    if hasattr(zoom, '__iter__'):
        raise Exception("zoom must be a single number!")
    if image.ndim == 1:
        # in this case, the two images were different sizes and so numpy can't combine them
        # correctly
        zooms, max_shape = find_zooms(image)
    elif image.ndim == 2:
        image = image.reshape((1, image.shape[0], image.shape[1]))
        max_shape = image.shape[1:]
        zooms = [1]
    else:
        zooms = [1 for i in image]
        max_shape = image.shape[1:]
    max_shape = np.array(max_shape)
    zooms = zoom * np.array(zooms)
    if not ((zoom * max_shape).astype(int) == zoom * max_shape).all():
        raise Exception("zoom * image.shape must result in integers!")

    if ax is None:
        if col_wrap is None:
            n_cols = image.shape[0]
            n_rows = 1
        else:
            n_cols = col_wrap
            n_rows = int(np.ceil(image.shape[0] / n_cols))
        if title is None:
            vert_pct = 1
        else:
            vert_pct = .8
        fig = make_figure(n_rows, n_cols, zoom * max_shape, vert_pct=vert_pct)
        axes = fig.axes
    else:
        fig = ax.figure
        axes = [reshape_axis(ax,  zoom * max_shape)]

    vrange_list, cmap = colormap_range(image=image, vrange=vrange, cmap=cmap)
    # print('passed', vrange_list)

    for im, a, r, t, z in zip(image, axes, vrange_list, title, zooms):
        # z in zooms
        _showIm(im, a, r, z, t, cmap, **kwargs)

    return fig


def animshow(movie, framerate=2., vrange='auto', zoom=1, as_html5=True, repeat=False, **kwargs):
    """Turn a 3D movie array into a matplotlib animation or HTML movie.

    Arguments
    ---------
    movie : `np.array` or `list`
        Array with time on the first axis or, equivalently, a list of 2d arrays. these 2d arrays
        don't have to all be the same size, but, if they're not, there must exist an integer such
        that all of them can be zoomed in by an integer up to the biggest image.
    framerate : `float`
        Temporal resolution of the movie, in Hz (frames per second).
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to the minimum and
        maximum value of the colormap, respectively. If a string:

        * `'auto/auto1'`: all images have same vmin/vmax, which are the minimum/maximum values
                          across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across all images) minus/
                     plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the 10th/90th percentile
                     values to the 10th/90th percentile of the display intensity range. For
                     example: vmin is the 10th percentile image value minus 1/8 times the
                     difference between the 90th and 10th percentile
        * `'indep1'`: each image has an independent vmin/vmax, which are their minimum/maximum
                      values
        * `'indep2'`: each image has an independent vmin/vmax, which is their mean minus/plus 2
                      std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that the 10th/90th
                      percentile values map to the 10th/90th percentile intensities.
    zoom : `float`
        amount we zoom the movie frames (must result in an integer when multiplied by
        movie.shape[1:])
    as_html : `bool`
        If True, return an HTML5 video; otherwise return the underying matplotlib animation object
        (e.g. to save to .gif). should set to True to display in a Jupyter notebook.
    repeat : `bool`
        whether to loop the animation or just play it once

    Returns
    -------
    anim : HTML object or FuncAnimation object
        Animation, format depends on `as_html`.

    """

    vrange_list, cmap = colormap_range(movie, vrange=vrange, cmap=kwargs.pop('cmap', None))

    _, max_shape = find_zooms(movie)
    max_shape = np.array(max_shape)
    if not ((zoom * max_shape).astype(int) == zoom * max_shape).all():
        raise Exception("zoom * movie.shape[1:] must result in integers!")
    # Initialize the figure and an empty array for the frames
    f = imshow(np.zeros(max_shape), zoom=zoom, vrange=vrange_list[0], cmap=cmap, title=None, **kwargs)
    array = f.axes[0].images[0]

    # Define animation functions
    def init_movie():
        return array,

    def animate_movie(i):
        frame = movie[i].astype(np.float)
        array.set_data(frame)
        array.set_clim(vrange_list[i])
        return array,

    # Produce the animation
    anim = animation.FuncAnimation(f, frames=len(movie), interval=1000/framerate, blit=True,
                                   func=animate_movie, init_func=init_movie, repeat=repeat,
                                   repeat_delay=500)

    plt.close(f)

    if as_html5:
        # to_html5_video will call savefig with a dpi kwarg, so our custom figure class will raise
        # a warning. we don't want to worry people, so we go ahead and suppress it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return HTML(anim.to_html5_video())
    return anim


def pyrshow(pyr_coeffs, is_complex=False, vrange='indep1', col_wrap=None, zoom=1, show_residuals=True, **kwargs):
    """Display the coefficients of the pyramid in an orderly fashion

    NOTE: this currently only works for 2d signals. we still need to figure out how to handle 1D
    signals.

    Arguments
    ---------
    pyr_coeffs : `dict`
        from the pyramid object (i.e. pyr.pyr_coeffs)
    is_complex : `bool`
        default False, indicates whether the pyramids is real or complex
        indicating whether the pyramid is complex or real
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to the minimum and
        maximum value of the colormap, respectively. If a string:

        * `'auto/auto1'`: all images have same vmin/vmax, which are the minimum/maximum values
                          across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across all images) minus/
                     plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the 10th/90th percentile
                     values to the 10th/90th percentile of the display intensity range. For
                     example: vmin is the 10th percentile image value minus 1/8 times the
                     difference between the 90th and 10th percentile
        * `'indep1'`: each image has an independent vmin/vmax, which are their minimum/maximum
                      values
        * `'indep2'`: each image has an independent vmin/vmax, which is their mean minus/plus 2
                      std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that the 10th/90th
                      percentile values map to the 10th/90th percentile intensities.
    col_wrap : `int` or None
        Only usable when the pyramid is one-dimensional (e.g., Gaussian or Laplacian Pyramid),
        otherwise the column wrap is determined by the number of bands. If not None, how many axes
        to have in a given row.
    zoom : `float`
        how much to scale the size of the images by. zoom times the size of the largest image must
        be an integer (and thus zoom should probably be an integer or 1/(2^n)).
    show_residuals : `bool`
        whether to display the residual bands (lowpass, highpass depending on the pyramid type)

    any additional kwargs will be passed through to imshow.

    Returns
    -------
    fig: `PyrFigure`
        the figure displaying the coefficients.
    """
    # right now, we do *not* do this the same as the old code. Instead of taking the coefficients
    # and arranging them in a spiral, we use imshow and arrange them neatly, displaying all at the
    # same size (and zoom / original image size clear), with different options for vrange. It
    # doesn't seem worth it to me to implement a version that looks like the old one, since that
    # would be a fair amount of work for dubious payoff, but if you want to implement that, the way
    # to do it is to probably use gridspec (https://matplotlib.org/users/gridspec.html), arranging
    # the axes in an orderly way and then passing them through to imshow somehow, rather than
    # pasting all coefficients into a giant array.
    # and the steerable pyramids have a num_orientations attribute

    # TODO make list of different elements in each dim
    # then only loop through those - see below line 655
    num_scales = np.max(np.array([k for k in pyr_coeffs.keys() if isinstance(k, tuple)])[:,0]) + 1
    num_orientations = np.max(np.array([k for k in pyr_coeffs.keys() if isinstance(k, tuple)])[:,1]) + 1
    # print(num_scales, num_orientations, zoom)

    col_wrap_new = num_orientations
    if is_complex:
        col_wrap_new *= 2
    # not sure about scope here, so we make sure to copy the
    # pyr_coeffs dictionary.
    imgs, highpass, lowpass = convert_pyr_coeffs_to_pyr(pyr_coeffs.copy())
    # we can similarly grab the labels for height and band
    # from the keys in this pyramid coefficients dictionary
    pyr_coeffs_keys = [k for k in pyr_coeffs.keys() if isinstance(k, tuple)]
    titles = ["height %02d, band %02d" % (h, b) for h, b in sorted(pyr_coeffs_keys)]
    if show_residuals:
        if highpass is not None:
            titles += ["residual highpass"]
            imgs.append(highpass)
        if lowpass is not None:
            titles += ["residual lowpass"]
            imgs.append(lowpass)
    if col_wrap_new is not None and col_wrap_new != 1:
        if col_wrap is None:
            col_wrap = col_wrap_new
    # if these are really 1d (i.e., have shape (1, x) or (x, 1)), then we want them to be 1d
    imgs = [i.squeeze() for i in imgs]
    if imgs[0].ndim == 1:
        # then we just want to plot each of the bands in a different subplot, no need to be fancy.
        if col_wrap is not None:
            warnings.warn("When the pyramid is 1d, we ignore col_wrap and just use "
                          "pyr.num_orientations to determine the number of columns!")
        height = num_scales
        if "residual highpass" in titles:
            height += 1
        if "residual lowpass" in titles:
            height += 1
        fig, axes = plt.subplots(height, num_orientations,
                                 figsize=(5*zoom, 5*zoom*num_orientations), **kwargs)
        plt.subplots_adjust(hspace=1.2, wspace=1.2)
        for i, ax in enumerate(axes.flatten()):
            ax.plot(imgs[i])
            ax.set_title(titles[i])
        return fig
    else:
        return imshow(imgs, vrange=vrange, col_wrap=col_wrap, zoom=zoom, title=titles, **kwargs)
