import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from IPython.display import HTML

# TODO subclass matplotlib figure so taht save fig uses correct dpi_scale_trans# ie. an integer multiple of the one used for creation of the figure
# return an error message on plt.tight_layout
# take a look at plt.saveim

def make_figure(n_rows, n_cols, axis_size_pix, col_margin_pix=10, row_margin_pix=10):

    # this is an arbitrary value
    ppi = 96

    # add extra 20% to the y direction for extra info
    # TODO if no title use all space
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
    if title is not None:
        # TODO adapt the precision of displayed range to the order of magnitude of the values: .1E
        ax.set_title(title + '\n range: [{:.1f}, {:.1f}] \n dims: [{}, {}] * {}'.format(
                     vrange[0], vrange[1], img.shape[0], img.shape[1], zoom), {'fontsize': ax.bbox.height*(12./256)})


def reshape_axis(ax, axis_size_pix):
    # NOTE - can only shrink a big ax, not blow up one that is too small

    if ax.bbox.width < axis_size_pix[1] or ax.bbox.height < axis_size_pix[0]:
        raise Exception("Your axis is too small! Axis size: ({}, {}). Image size: ({}, {})".format(ax.bbox.width, ax.bbox.height, axis_size_pix[1], axis_size_pix[0]))
    bbox = ax.figure.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*ax.figure.dpi, bbox.height*ax.figure.dpi
    rel_axis_width = axis_size_pix[1] / fig_width
    rel_axis_height = axis_size_pix[0] / fig_height
    ax.set_position([*ax.get_position().bounds[:2], rel_axis_width, rel_axis_height])
    return ax


def showIm(img, vrange=None, zoom=1, title='', col_wrap=None, ax=None,
            cmap=cm.gray, **kwargs):
    '''under construction

    img: 2d array (one image to display), 3d array (multiple images to display,
    images are indexed along the first dimension), or list of 2d arrays.

    vrange: None/auto
    auto2
    auto3

    zoom

    title: string or list of strings. if string, will put the same title on
    every plot. if list of strings, must be the same length as img, and will
    assume that titles go with the corresponding image. If title is None,
    no title will be printed

    ax: matplotlib axis or None. If None, make the appropriate figure.
    If not None, we reshape it (which we only do by shrinking the bbox,
    so if the bbox is already too small, this will throw an Exception!)
    so that it's the appropriate number of pixels.
    first define a large enough figure using either make_figure or plt.figure

    col_wrap:

    TODO:
    independent vrrange for subimages

    diff zoom values


    low priority: nshades

    '''

    img = np.array(img)

    if img.ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
    # if not isinstance(zoom, list):
    #     zoom = len(img) * [zoom]
    # else:
    #     assert len(img) == len(zoom), "Must have same number of zooms and images!"
    if ax is None:
        if col_wrap is None:
            n_cols = img.shape[0]
            n_rows = 1
        else:
            n_cols = col_wrap
            n_rows = int(np.ceil(img.shape[0] / n_cols))
        # TODO zoom list
        fig = make_figure(n_rows, n_cols, zoom * np.array(img.shape[1:]))
        axes = fig.axes
    else:
        fig = ax.figure
        axes = [reshape_axis(ax, img.shape[1:])]
    if not isinstance(title, list):
        title = len(img) * [title]
    else:
        assert len(img) == len(title), "Must have same number of titles and images!"

    # TODO modularize - put in a different function
    if vrange is None or vrange == 'auto':
        vrange = [np.min(img), np.max(img)]
    # TODO add option to keep vrange independent accross sub images
    elif vrange == 'auto2':
        vrange = [img.mean() - 2 * img.std(),
                  img.mean() + 2 * img.std()]
    elif vrange == 'auto3':
        p1 = np.percentile(img, 10)
        p2 = np.percentile(img, 90)
        vrange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)
    # add error message if vrange not one of these


    for a, im, t in zip(axes, img, title): # add zoom
        _showIm(im, a, vrange, zoom, t, cmap, **kwargs)

    return fig

def animshow(movie, framerate=1 / 60, vrange= None, size=5, as_html5=True,
               **kwargs):
    """Turn a 3D movie array into a matplotlib animation or HTML movie.

    Parameters
    ----------
    movie : 3D numpy array
        Array with time on the final axis.
    framerate : float
        Temporal resolution of the movie, in frames per second.
    aperture : bool
        If True, show only a central circular aperture.
    size : float
        Size of the underlying matplotlib figure, in inches.
    as_html : bool
        If True, return an HTML5 video; otherwise return the underying
        matplotlib animation object (e.g. to save to .gif).

    Returns
    -------
    anim : HTML object or FuncAnimation object
        Animation, format depends on `as_html`.

    TODO
    size -> zoom, control ppi (reuse previous showIm functions?)

    """

    # TODO modularize the vrange function and use it in both imshow and animshow
    if vrange is None or vrange == 'auto':
        vrange = [np.min(movie), np.max(movie)]
    elif vrange == 'auto2':
        vrange = [movie.mean() - 2 * movie.std(),
                  movie.mean() + 2 * movie.std()]
    elif vrange == 'auto3':
        p1 = np.percentile(movie, 10)
        p2 = np.percentile(movie, 90)
        vrange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)

    # Initialize the figure and an empty array for the frames
    f, ax = plt.subplots(figsize=(size, size))
    f.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()

    kwargs.setdefault("vmin", vrange[0])
    kwargs.setdefault("vmax", vrange[1])
    kwargs.setdefault("cmap", "gray")
    array = ax.imshow(np.zeros(movie.shape[:-1]), **kwargs)

    # Define animation functions
    def init_movie():
        return array,

    def animate_movie(i):
        frame = movie[..., i].astype(np.float)
        array.set_data(frame)
        return array,

    # Produce the animation
    anim = animation.FuncAnimation(f,
                                   frames=movie.shape[-1],
                                   interval=framerate * 1000,
                                   blit=True,
                                   func=animate_movie,
                                   init_func=init_movie)

    plt.close(f)

    if as_html5:
        return HTML(anim.to_html5_video())
    return anim
