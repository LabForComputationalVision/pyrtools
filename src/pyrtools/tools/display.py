import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib import animation
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
        the image(s) to be shown. should be a 2d array (one image to display),
        3d array (multiple images to display, images are indexed along the first
        dimension), or list of 2d arrays. all images will be automatically
        rescaled so they're displayed at the same size. thus, their sizes must
        be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        (ie. clipped to) the minimum and maximum value of the colormap,
        respectively.
        If a string:
        * `'auto0'`: all images have same vmin/vmax, which have the same
                     absolute value, and come from the minimum or maximum across
                     all images, whichever has the larger absolute value
        * `'auto1'`: all images have same vmin/vmax, which are the minimum/
                     maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                     same absolute value, which comes from either their minimum
                     or maximum value, whichever has the larger absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                     minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                     mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                     the 10th/90th percentile values map to the 10th/90th
                     percentile intensities.

    Returns
    -------
    vrange_list : `list`
        list of tuples, same length as `image`. contains the (vmin, vmax) tuple
        for each image.

    """
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
                vrange_list = [np.nanmean(flatimg) - 2 * np.nanstd(flatimg),
                               np.nanmean(flatimg) + 2 * np.nanstd(flatimg)]
            elif vrange == 'auto3':
                p1 = np.nanpercentile(flatimg, 10)
                p2 = np.nanpercentile(flatimg, 90)
                vrange_list = [p1-(p2-p1)/8.0, p2+(p2-p1)/8.0]

            # make sure to return as many ranges as there are images
            vrange_list = [vrange_list] * len(image)

        elif vrange[:5] == 'indep':
            # independent vrange from recursive calls of this function per image
            vrange_list = [colormap_range(im, vrange.replace('indep', 'auto')
                           )[0][0] for im in image]
        else:
            vrange_list, _ = colormap_range(image, vrange='auto1')
            warnings.warn('Unknown vrange argument, using auto1 instead')
    else:
        # two numbers were passed, either as a list or tuple
        if len(vrange) != 2:
            raise Exception("If you're passing numbers to vrange,"
                            "there must be 2 of them!")
        vrange_list = [tuple(vrange)] * len(image)

    # double check that we're returning the right number of vranges
    assert len(image) == len(vrange_list)

    if cmap is None:
        if '0' in vrange:
            cmap = cm.RdBu_r
        else:
            cmap = cm.gray

    return vrange_list, cmap


def _check_shapes(images, video=False):
    """Helper function to check whether images can be zoomed in appropriately.

    this convenience function takes a list of images and finds out if they can all be displayed at
    the same size. for this to be the case, there must be an integer for each image such that the
    image can be multiplied by that integer to be the same size as the biggest image.

    Arguments
    ---------
    images : `list`
        list of numpy arrays to check the size of. In practice, these are 1d or 2d, but can in
        principle be any number of dimensions
    video: bool, optional (default False)
        handling signals in both space and time or only space.

    Returns
    -------
    max_shape : `tuple`
        2-tuple of integers, showing the shape of the largest image in the list

    Raises
    ------
    ValueError :
        If the images cannot be zoomed to the same. that is, if there is not an integer
        for each image such that the image can be multiplied by that integer to be the
        same size as the biggest image.
    """
    def check_shape_1d(shapes):
        max_shape = np.max(shapes)
        for s in shapes:
            if not (max_shape % s) == 0:
                raise ValueError("All images must be able to be 'zoomed in' to the largest image."
                                 "That is, the largest image must be a scalar multiple of all "
                                 "images.")
        return max_shape
    if video:
        time_dim = 1
    else:
        time_dim = 0
    max_shape = []
    for i in range(2):
        max_shape.append(check_shape_1d([img.shape[i+time_dim] for img in images]))
    return max_shape


def find_zooms(images, video=False):
    """find the zooms necessary to display a list of images

    Arguments
    ---------
    images : `list`
        list of numpy arrays to check the size of. In practice, these are 1d or 2d, but can in
        principle be any number of dimensions
    video: bool, optional (default False)
        handling signals in both space and time or only space.

    Returns
    -------
    zooms : `list`
        list of integers showing how much each image needs to be zoomed
    max_shape : `tuple`
        2-tuple of integers, showing the shape of the largest image in the list

    Raises
    ------
    ValueError :
        If the images cannot be zoomed to the same. that is, if there is not an integer
        for each image such that the image can be multiplied by that integer to be the
        same size as the biggest image.
    ValueError :
        If the two image dimensions require different levels of zoom (e.g., if the
        height must be zoomed by 2 but the width must be zoomed by 3).

    """
    max_shape = _check_shapes(images, video)
    if video:
        time_dim = 1
    else:
        time_dim = 0
    zooms = []
    for img in images:
        # this checks that there's only one unique value in the list
        # max_shape[i] // img.shape[i], where i indexes through the dimensions;
        # that is, that we zoom each dimension by the same amount. this should
        # then work with an arbitrary number of dimensions (in practice, 1 or
        # 2). by using max_shape instead of img.shape, we will only ever check
        # the first two non-time dimensions (so we'll ignore the RGBA channel
        # if any image has that)
        if len(set([s // img.shape[i+time_dim] for i, s in enumerate(max_shape)])) > 1:
            raise ValueError("Both height and width must be multiplied by same amount but got "
                             "image shape {} and max_shape {}!".format(img.shape, max_shape))
        zooms.append(max_shape[0] // img.shape[0])
    return zooms, max_shape


def _convert_signal_to_list(signal):
    """Convert signal to list.

    signal can be an array or a list of arrays. this guarantees it's a list,
    and raises an Exception if it's not an array or list.

    if it's an already a list, we don't check whether each value is a
    properly-shaped array, nor do we check the shape or dimensionality of a
    single array. these both happen in _process_signal

    Parameters
    ----------
    signal : np.ndarray or list
        the array or list of arrays to convert

    Returns
    -------
    signal : list

    """
    try:
        if isinstance(signal, np.ndarray):
            # then this is a single signal
            signal = [signal]
    except AttributeError:
        # then this is a list and we don't do anything
        pass
    if not isinstance(signal, list):
        raise TypeError("image must be a np.ndarray or a list! {} is unsupported".format(type(signal)))
    return signal


def _convert_title_to_list(title, signal):
    """Convert title to list and get vert_pct.

    This function makes sure that title is a list with the right number of
    elements, and sets vert_pct based on whether title is None or not

    Parameters
    ----------
    title : str, list, or None
        the title to check
    signal : list
        the signal (e.g., images) that will be plotted. must already have gone
        through _convert_signal_to_list

    Returns
    -------
    title : list
        list of title
    vert_pct : float
        how much of the axis should contain the image

    """
    if title is None:
        vert_pct = 1
    else:
        vert_pct = .8
    if not isinstance(title, list):
        title = len(signal) * [title]
    else:
        assert len(signal) == len(title), "Must have same number of titles and images!"
    return title, vert_pct


def _process_signal(signal, title, plot_complex, video=False):
    """Process signal and title for plotting.

    Two goals of this function:

    1. Check the shape of each image to make sure they look like either
       grayscale or RGB(A) images and raise an Exception if they don't

    2. Process complex images for proper plotting, splitting them up
       as specified by `plot_complex`

    Parameters
    ----------
    signal : list
        list of arrays to examine
    title : list
        list containing strs or Nones, for accompanying the images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        how to plot complex arrays
    video: bool, optional (default False)
        handling signals in both space and time or only space.

    Returns
    -------
    signal : list
        list of arrays containing the signal, ready to plot
    title : list
        list of titles, ready to plot
    contains_rgb : bool
        if at least one of the images is 3d (and thus RGB), this will be True.

    """
    if video:
        time_dim = 1
        sigtype = 'video'
    else:
        time_dim = 0
        sigtype = 'image'
    signal_tmp = []
    title_tmp = []
    contains_rgb = False
    for sig, t in zip(signal, title):
        if sig.ndim == (3 + time_dim):
            if sig.shape[-1] not in [3, 4]:
                raise Exception(
                    f"Can't figure out how to plot {sigtype} with shape {sig.shape} "
                    "as RGB(A)! RGB(A) signals should have their final"
                    "dimension of shape 3 or 4."
                )
            contains_rgb = True
        elif sig.ndim != (2 + time_dim):
            raise Exception(
                f"Can't figure out how to plot {sigtype} with "
                f"shape {sig.shape}! {sigtype.capitalize()}s should be be either "
                f"{2 + time_dim}d (grayscale) or {3 + time_dim}d (RGB(A), last "
                "dimension with 3 or 4 elements)."
            )
        if np.iscomplexobj(sig):
            if plot_complex == 'rectangular':
                signal_tmp.extend([np.real(sig), np.imag(sig)])
                if t is not None:
                    title_tmp.extend([t + " real", t + " imaginary"])
                else:
                    title_tmp.extend([None, None])
            elif plot_complex == 'polar':
                signal_tmp.extend([np.abs(sig), np.angle(sig)])
                if t is not None:
                    title_tmp.extend([t + " amplitude", t + " phase"])
                else:
                    title_tmp.extend([None, None])
            elif plot_complex == 'logpolar':
                signal_tmp.extend([np.log(1 + np.abs(sig)), np.angle(sig)])
                if t is not None:
                    title_tmp.extend([t + " log(1+amplitude)", t + " phase"])
                else:
                    title_tmp.extend([None, None])
        else:
            signal_tmp.append(np.asarray(sig))
            title_tmp.append(t)
    return signal_tmp, title_tmp, contains_rgb


def _check_zooms(signal, zoom, contains_rgb, video=False):
    """Check that all images can be zoomed correctly.

    Make sure that all images can be zoomed so they end up the same size, and
    figure out how to do that

    Parameters
    ----------
    signal : list
        list of arrays that will be plotted
    zoom : float
        how we're going to zoom the image
    contains_rgb : bool
        whether image contains at least one image to plot as RGB. This only
        matters when we're given a 3d array and we want to know whether it was
        supposed to be a single RGB image or multiple grayscale ones
    video: bool, optional (default False)
        handling signals in both space and time or just space.

    Returns
    -------
    zooms : np.ndarray
        how much to zoom each image
    max_shape : np.ndarray
        contains 2 ints, giving the max image size in pixels

    """
    if video:
        time_dim = 1
    else:
        time_dim = 0

    if hasattr(zoom, '__iter__'):
        raise Exception("zoom must be a single number!")
    if all([sig.shape == signal[0].shape for sig in signal]):
        # then this is one or multiple images/videos, all the same size
        zooms = [1 for i in signal]
        if contains_rgb:
            max_shape = signal[0].shape[-3:-1]
        else:
            max_shape = signal[0].shape[-2:]
    else:
        # then we have multiple images/videos that are different shapes
        zooms, max_shape = find_zooms(signal, video)
    max_shape = np.asarray(max_shape)
    zooms = zoom * np.asarray(zooms)
    if not ((zoom * max_shape).astype(int) == zoom * max_shape).all():
        raise Exception("zoom * signal.shape must result in integers!")
    return zooms, max_shape


def _setup_figure(ax, col_wrap, image, zoom, max_shape, vert_pct):
    """Create figure with appropriate arrangement and size of axes

    Creates (or tries to resize) set of axes for the appropriate arguments

    """
    if ax is None:
        if col_wrap is None:
            n_cols = len(image)
            n_rows = 1
        else:
            n_cols = col_wrap
            n_rows = int(np.ceil(len(image) / n_cols))
        fig = make_figure(n_rows, n_cols, zoom * max_shape, vert_pct=vert_pct)
        axes = fig.axes
    else:
        fig = ax.figure
        axes = [reshape_axis(ax,  zoom * max_shape)]
    return fig, axes


def imshow(image, vrange='indep1', zoom=1, title='', col_wrap=None, ax=None,
           cmap=None, plot_complex='rectangular', **kwargs):
    """Show image(s).

    Arguments
    ---------
    image : `np.array` or `list`
        the image(s) to plot. Images can be either grayscale, in which case
        they must be 2d arrays of shape `(h,w)`, or RGB(A), in which case they
        must be 3d arrays of shape `(h,w,c)` where `c` is 3 (for RGB) or 4 (to
        also plot the alpha channel). If multiple images, must be a list of
        such arrays (note this means we do not support an array of shape
        `(n,h,w)` for multiple grayscale images). all images will be
        automatically rescaled so they're displayed at the same size. thus,
        their sizes must be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image.
    title : `str`, `list`, or None, optional
        Title for the plot. In addition to the specified title, we add a
        subtitle giving the plotted range and dimensionality (with zoom)

        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img,
          assigning each title to corresponding image.
        * if `None`, no title will be printed (and subtitle will be removed;
          unsupported for complex tensors).
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}, optional
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot `ln(1+ amplitude)` and phase as separate images.
        The compressive non-linear contrast function applied to amplitude is
        intended as a visualization step to avoid the large intensity
        components from dominating.
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    fig : `PyrFigure`
        figure containing the plotted images

    """
    if plot_complex not in ['rectangular', 'polar', 'logpolar']:
        raise Exception("Don't know how to handle plot_complex value "
                        "{}!".format(plot_complex))

    # Make sure image is a list, do some preliminary checks
    image = _convert_signal_to_list(image)

    # want to do this check before converting title to a list (at which
    # point `title is None` will always be False). we do it here instad
    # of checking whether the first item of title is None because it's
    # conceivable that the user passed `title=[None, 'important
    # title']`, and in that case we do want the space for the title
    title, vert_pct = _convert_title_to_list(title, image)

    # Process complex images for plotting, double-check image size to see if we
    # have RGB(A) images
    image, title, contains_rgb = _process_signal(image, title, plot_complex)

    # make sure we can properly zoom all images
    zooms, max_shape = _check_zooms(image, zoom, contains_rgb)

    # get the figure and axes created
    fig, axes = _setup_figure(ax, col_wrap, image, zoom, max_shape, vert_pct)

    vrange_list, cmap = colormap_range(image=image, vrange=vrange, cmap=cmap)

    for im, a, r, t, z in zip(image, axes, vrange_list, title, zooms):
        _showIm(im, a, r, z, t, cmap, **kwargs)

    return fig


def animshow(video, framerate=2., as_html5=True, repeat=False,
             vrange='indep1', zoom=1, title='', col_wrap=None, ax=None,
             cmap=None, plot_complex='rectangular', **kwargs):
    """Display one or more videos (3d array) as a matplotlib animation or an HTML video.

    Arguments
    ---------
    video : `np.array` or `list`
        the videos(s) to show. Videos can be either grayscale, in which case
        they must be 3d arrays of shape `(f,h,w)`, or RGB(A), in which case
        they must be 4d arrays of shape `(f,h,w,c)` where `c` is 3 (for RGB) or
        4 (to also plot the alpha channel) and `f` indexes frames. If multiple
        videos, must be a list of such arrays (note this means we do not
        support an array of shape `(n,f,h,w)` for multiple grayscale videos).
        all videos will be automatically rescaled so they're displayed at the
        same size. thus, their sizes must be scalar multiples of each other. If
        multiple videos, all must have the same number of frames (first
        dimension).
    framerate : `float`
        Temporal resolution of the video, in Hz (frames per second).
    as_html : `bool`
        If True, return an HTML5 video; otherwise return the underying matplotlib animation object
        (e.g. to save to .gif). should set to True to display in a Jupyter notebook.
        Requires ipython to be installed.
    repeat : `bool`
        whether to loop the animation or just play it once
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
        amount we zoom the video frames (must result in an integer when multiplied by
        video.shape[1:])
    title : `str` , `list` or None
        Title for the plot:

        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img, assigning each
          title to corresponding image.
        * if None, no title will be printed.
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}, optional
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    anim : HTML object or FuncAnimation object
        Animation, format depends on `as_html`.

    """
    if as_html5:
        try:
            from IPython.display import HTML
        except ImportError:
            raise ImportError("Unable to import IPython.display.HTML, animshow must be called with "
                              "as_html5=False")
    video = _convert_signal_to_list(video)
    video_n_frames = np.asarray([v.shape[0] for v in video])
    if (video_n_frames != video_n_frames[0]).any():
        raise Exception("All videos must have the same number of frames! But you "
                        "passed videos with {} frames".format(video_n_frames))
    title, vert_pct = _convert_title_to_list(title, video)
    video, title, contains_rgb = _process_signal(video, title, plot_complex, video=True)
    zooms, max_shape = _check_zooms(video, zoom, contains_rgb, video=True)
    fig, axes = _setup_figure(ax, col_wrap, video, zoom, max_shape, vert_pct)
    vrange_list, cmap = colormap_range(image=video, vrange=vrange, cmap=cmap)

    first_image = [v[0] for v in video]
    for im, a, r, t, z in zip(first_image, axes, vrange_list, title, zooms):
        _showIm(im, a, r, z, t, cmap, **kwargs)

    artists = [fig.axes[i].images[0] for i in range(len(fig.axes))]

    for i, a in enumerate(artists):
        a.set_clim(vrange_list[i])

    def animate_video(t):
        for i, a in enumerate(artists):
            frame = video[i][t].astype(float)
            a.set_data(frame)
        return artists

    # Produce the animation
    anim = animation.FuncAnimation(fig, frames=len(video[0]),
                                   interval=1000/framerate, blit=True,
                                   func=animate_video, repeat=repeat,
                                   repeat_delay=500)

    plt.close(fig)

    if as_html5:
        # to_html5_video will call savefig with a dpi kwarg, so our custom figure class will raise
        # a warning. we don't want to worry people, so we go ahead and suppress it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return HTML(anim.to_html5_video())
    return anim


def pyrshow(pyr_coeffs, is_complex=False, vrange='indep1', col_wrap=None, zoom=1, show_residuals=True, **kwargs):
    """Display the coefficients of the pyramid in an orderly fashion

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

    num_scales = np.max(np.asarray([k for k in pyr_coeffs.keys() if isinstance(k, tuple)])[:,0]) + 1
    num_orientations = np.max(np.asarray([k for k in pyr_coeffs.keys() if isinstance(k, tuple)])[:,1]) + 1

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
        try:
            _check_shapes(imgs)
        except ValueError:
            err_scales = num_scales
            residual_err_msg = ""
            shapes = [(imgs[0].shape[0]/ 2**i, imgs[0].shape[1] / 2**i) for i in range(err_scales)]
            err_msg = [f"scale {i} shape: {sh}" for i, sh in enumerate(shapes)]
            if show_residuals:
                err_scales += 1
                residual_err_msg = ", plus 1 (for the residual lowpass)"
                shape = (imgs[0].shape[0]/ int(2**err_scales), imgs[0].shape[1] / int(2**err_scales))
                err_msg.append(f"residual lowpass shape: {shape}")
            err_msg = "\n".join(err_msg)
            raise ValueError("In order to correctly display pyramid coefficients, the shape of"
                             f" the initial image must be evenly divisible by two {err_scales} "
                             "times, where this number is the height of the "
                             f"pyramid{residual_err_msg}. "
                             f"Instead, found:\n{err_msg}")
        return imshow(imgs, vrange=vrange, col_wrap=col_wrap, zoom=zoom, title=titles, **kwargs)
