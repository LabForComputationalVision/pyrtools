class struct( object ):
    def __init__( self, **kwargs ):
        for k, v in kwargs.iteritems():
            setattr( self, k, v )


class progressbar:
    def __init__( self ):
        from uuid import uuid4
        from IPython.display import display, HTML
        from time import time
        self.divid = str( uuid4() )
        self.last_update = time()
        self.last_perc = 0
        pb = HTML( '<div style="background-color: #DDDDDD; border-radius: 4px; padding: 1px; width: 500px"><div id="{}" style="background-color: #F7F7F7; height: 14px; border-radius: 3px; width: 0%"></div></div>'.format( self.divid ) )
        display( pb )
    def update( self, percentage, minsecs = None ):
        from IPython.display import display, Javascript
        from time import time
        now = time()
        if percentage >= 100:
            percentage = 100
        if minsecs is None or now - self.last_update > minsecs or ( self.last_perc < 100 and percentage == 100 ):
            display( Javascript( "$('div#{}').width('{}%')".format( self.divid, int( percentage ) ) ) )
            self.last_update = now
            self.last_perc = percentage


def rerange( data, vmin = None, vmax = None, vsym = False ):
    '''
    Rescale values of data array to fit the range 0 ... 256 and convert to uint8.

    Parameters:
    data: array-like object. if data.dtype == uint8, no scaling will occur.
    vmin: original array value that will map to 0 in the output. [ data.min() ]
    vmax: original array value that will map to 255 in the output. [ data.max() ]
    vsym: ensure that 0 will map to gray (if True, may override either vmin or vmax
          to accommodate all values.) [ False ]
    '''
    from numpy import asarray, uint8, clip
    data = asarray( data )
    if data.dtype != uint8:
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        if vsym:
            vmax = max( abs( vmin ), abs( vmax ) )
            vmin = -vmax
        data = ( data - vmin ) * ( 256 / ( vmax - vmin ) )
    return clip( data, 0, 255 ).astype( uint8 )


def nbimage( data, vmin = None, vmax = None, vsym = False, saveas = None ):
    '''
    Display raw data as a notebook inline image.

    Parameters:
    data: array-like object, two or three dimensions. If three dimensional,
          first or last dimension must have length 3 or 4 and will be
          interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    saveas: Save image file to disk (optional). Proper file name extension
            will be appended to the pathname given. [ None ]
    '''
    from IPython.display import display, Image
    from PIL.Image import fromarray
    from StringIO import StringIO
    data = rerange( data, vmin, vmax, vsym )
    data = data.squeeze()
    # try to be smart
    if 3 <= data.shape[ 0 ] <= 4:
        data = data.transpose( ( 1, 2, 0 ) )
    s = StringIO()
    fromarray( data ).save( s, 'png' )
    if saveas is not None:
        open( saveas + '.png', 'wb' ).write( s )
    display( Image( s.getvalue() ) )

def nbimageLCVbak( data, vmin = None, vmax = None, vsym = False, saveas = None, 
                zoom = 1 ):
    '''
    Display raw data as a notebook inline image.

    Parameters:
    data: array-like object, two or three dimensions. If three dimensional,
          first or last dimension must have length 3 or 4 and will be
          interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    saveas: Save image file to disk (optional). Proper file name extension
            will be appended to the pathname given. [ None ]
    zoom: amount to scale the image
    '''
    from IPython.display import display, Image, HTML
    from PIL.Image import fromarray
    from StringIO import StringIO
    css_styling()
    data = rerange( data, vmin, vmax, vsym )
    data = data.squeeze()

    # try to be smart
    if 3 <= data.shape[ 0 ] <= 4:
        print 'transposing'
        data = data.transpose( ( 1, 2, 0 ) )

    s = StringIO()
    fromarray( data ).save( s, 'png' )
    if saveas is not None:
        open( saveas + '.png', 'wb' ).write( s )

    display( Image( s.getvalue(),width=data.shape[0]*zoom, 
                    height=data.shape[1]*zoom ) )
    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    html = 'Range: [%.1f, %.1f]  Dims: [%d, %d]*%.2f' % (vmin, vmax, 
                                                         data.shape[0],
                                                         data.shape[1], zoom)
    display( HTML( html ) )

def nbimageLCVbak2( data, vmin = None, vmax = None, vsym = False, saveas = None,
                    zoom = 1, nshades = 256, subDim = (1,1) ):
    '''
    Display raw data as a notebook inline image.

    Parameters:
    data: array-like object, two or three dimensions. If three dimensional,
          first or last dimension must have length 3 or 4 and will be
          interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    saveas: Save image file to disk (optional). Proper file name extension
            will be appended to the pathname given. [ None ]
    zoom: amount to scale the image
    '''
    from IPython.display import display, Image, HTML
    from PIL.Image import fromarray
    from StringIO import StringIO
    import base64
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    import numpy

    data = rerange( data, vmin, vmax, vsym )
    data = data.squeeze()
    # try to be smart
    if 3 <= data.shape[ 0 ] <= 4:
        print 'transposing'
        data = data.transpose( ( 1, 2, 0 ) )
    s = StringIO()
    fromarray( data ).save( s, 'png' )
    if saveas is not None:
        open( saveas + '.png', 'wb' ).write( s )

    #display( Image( s.getvalue(),width=data.shape[0]*zoom, 
    #                height=data.shape[1]*zoom ) )

    matrix = numpy.require(data, numpy.uint8, 'C')
    (w, h) = matrix.shape
    print matrix
    qim = QtGui.QImage(matrix.data, w, h, QtGui.QImage.Format_Indexed8)
    qim.ndarray = matrix    # do we need this?
    
    # make colormap
    incr = (256/nshades)+1
    colors = range(0,255,(256/nshades)+1)
    colors[-1] = 255
    colctr = -1
    for i in range(256):
        if i % incr == 0:
            colctr += 1
        qim.setColor(i, QtGui.QColor(colors[colctr], colors[colctr], 
                                     colors[colctr]).rgb())

    # zoom
    dims = (matrix.shape[0]*zoom, matrix.shape[1]*zoom)
    qim = qim.scaled(dims[0], dims[1])

    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.WriteOnly)
    qim.save(buf, 'PNG')
    base64_data = ba.toBase64().data()
    im = 'data:image/png;base64,' + base64_data
    head = """<table border=0><tr><th><center>"""
    foot = """</center></th></tr></table>"""
    s = """<img src="%s"/>"""%(im)

    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    html = 'Range: [%.1f, %.1f]<br>Dims: [%d, %d]*%.2f' % (vmin, vmax, 
                                                           data.shape[0],
                                                           data.shape[1], zoom)
    display( HTML ( head + s + html + foot ) )

    #display( HTML( html ) )

def nbimageLCV( dlist, vmin = None, vmax = None, vsym = False, saveas = None, 
                zoom = 1, nshades = 256, ncols = 1, title = ""):
    '''
    Display raw data as a notebook inline image.

    Parameters:
    data: array-like object, two or three dimensions. If three dimensional,
          first or last dimension must have length 3 or 4 and will be
          interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    saveas: Save image file to disk (optional). Proper file name extension
            will be appended to the pathname given. [ None ]
    zoom: amount to scale the image
    nshades: number of shades of grey
    ncols: number of columns of display images (subplotting)
    title: optional figure title
    '''
    from IPython.display import display, Image, HTML
    from PIL.Image import fromarray
    from StringIO import StringIO
    import base64
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    import numpy

    if not isinstance(dlist, list):
        dlist = [dlist]

    displayColCtr = 0
    imgCtr = -1
    s = ""
    s += "<table border=0>"
    for data in dlist:
        imgCtr += 1

        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        data = rerange( data, vmin, vmax, vsym )
        data = data.squeeze()
        # try to be smart
        if 3 <= data.shape[ 0 ] <= 4:
            data = data.transpose( ( 1, 2, 0 ) )
        #s = StringIO()
        #fromarray( data ).save( s, 'png' )
        #if saveas is not None:
        #    open( saveas + '.png', 'wb' ).write( s )
        # Thank you Johannes for the following two line fix!!!
        matrix = numpy.empty( ( data.shape[ 0 ], 
                                ( data.shape[ 1 ] + 3 ) // 4 * 4 ), 
                              numpy.uint8 )
        matrix[ :, :data.shape[ 1 ] ] = data
        (h, w) = data.shape
        qim = QtGui.QImage(matrix.data, w, h, QtGui.QImage.Format_Indexed8)
    
        # make colormap
        incr = (256/nshades)+1
        colors = range(0,255,(256/nshades)+1)
        colors[-1] = 255
        colctr = -1
        for i in range(256):
            if i % incr == 0:
                colctr += 1
            qim.setColor(i, QtGui.QColor(colors[colctr], colors[colctr], 
                                         colors[colctr]).rgb())

        # zoom
        if isinstance(zoom, list):
            currZoom = zoom[imgCtr]
        else:
            currZoom = zoom
        
        #dims = (matrix.shape[0]*zoom, matrix.shape[1]*zoom)
        #dims = (matrix.shape[0]*currZoom, matrix.shape[1]*currZoom)
        dims = (matrix.shape[1]*currZoom, matrix.shape[0]*currZoom)
        qim = qim.scaled(dims[0], dims[1])

        ba = QtCore.QByteArray()
        buf = QtCore.QBuffer(ba)
        buf.open(QtCore.QIODevice.WriteOnly)
        #im.save(buf, 'PNG')
        qim.save(buf, 'PNG')
        base64_data = ba.toBase64().data()

        if displayColCtr == 0:
            s += "<tr>"

        im = 'data:image/png;base64,' + base64_data

        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        #label = 'Range: [%.1f, %.1f]<br>Dims: [%d, %d]*%.2f' % (vmin, vmax, data.shape[0], data.shape[1], zoom)
        label = 'Range: [%.1f, %.1f]<br>Dims: [%d, %d]*%.2f' % (vmin, vmax, data.shape[0], data.shape[1], currZoom)
        #s += """<th><center><img src="%s"/>%s</center></th>"""%(im,label)
        #if title != "":
        #    title += "<br>"
        s += """<th><center>%s<img src="%s"/>%s</center></th>"""%(title,im,label)
        displayColCtr += 1
        if displayColCtr >= ncols:
            displayColCtr = 0
            s += "</tr>"

    s += "</table>"
    display( HTML ( s ) )

def showIm( dlist, v = None, zoom = 1, title = "", nshades = 256, ncols = 1):
    vsym = False
    saveas = None
    '''
    Display raw data as a notebook inline image.

    Parameters:
    data: array-like object, two or three dimensions. If three dimensional,
          first or last dimension must have length 3 or 4 and will be
          interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    saveas: Save image file to disk (optional). Proper file name extension
            will be appended to the pathname given. [ None ]
    zoom: amount to scale the image
    title: optional figure title
    nshades: number of shades of grey
    ncols: number of columns of display for images (subplotting)
    '''
    from IPython.display import display, Image, HTML
    from PIL.Image import fromarray
    from StringIO import StringIO
    import base64
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    import numpy

    if not isinstance(dlist, list):
        dlist = [dlist]

    displayColCtr = 0
    imgCtr = -1
    s = ""
    s += "<table border=0>"
    for data in dlist:
        imgCtr += 1

        if v == None or v == 'auto':
            vmin = data.min()
            vmax = data.max()
        elif v == 'auto2':
            vmin = matrix.mean()-2*matrix.std()
            mmax = matrix.mean()+2*matrix.std()
        elif v == "auto3":
            p1 = sps.scoreatpercentile(np.hstack(matrix), 10)
            p2 = sps.scoreatpercentile(np.hstack(matrix), 90)
            vmin = p1-(p2-p1)/8.0
            vmax = p2+(p2-p1)/8.0
        else:
            print "Error: range of %s is not recognized." % v
            print "       please use a two element tuple or "
            print "       'auto', 'auto2' or 'auto3'"
            return

        data = rerange( data, vmin, vmax, vsym )
        data = data.squeeze()
        # try to be smart
        if 3 <= data.shape[ 0 ] <= 4:
            data = data.transpose( ( 1, 2, 0 ) )
        #s = StringIO()
        #fromarray( data ).save( s, 'png' )
        #if saveas is not None:
        #    open( saveas + '.png', 'wb' ).write( s )
        # Thank you Johannes for the following two line fix!!!
        matrix = numpy.empty( ( data.shape[ 0 ], 
                                ( data.shape[ 1 ] + 3 ) // 4 * 4 ), 
                              numpy.uint8 )
        matrix[ :, :data.shape[ 1 ] ] = data
        (h, w) = data.shape
        qim = QtGui.QImage(matrix.data, w, h, QtGui.QImage.Format_Indexed8)
    
        # make colormap
        incr = (256/nshades)+1
        colors = range(0,255,(256/nshades)+1)
        colors[-1] = 255
        colctr = -1
        for i in range(256):
            if i % incr == 0:
                colctr += 1
            qim.setColor(i, QtGui.QColor(colors[colctr], colors[colctr], 
                                         colors[colctr]).rgb())

        # zoom
        if isinstance(zoom, list):
            currZoom = zoom[imgCtr]
        else:
            currZoom = zoom
        
        #dims = (matrix.shape[0]*zoom, matrix.shape[1]*zoom)
        #dims = (matrix.shape[0]*currZoom, matrix.shape[1]*currZoom)
        dims = (matrix.shape[1]*currZoom, matrix.shape[0]*currZoom)
        qim = qim.scaled(dims[0], dims[1])

        ba = QtCore.QByteArray()
        buf = QtCore.QBuffer(ba)
        buf.open(QtCore.QIODevice.WriteOnly)
        #im.save(buf, 'PNG')
        qim.save(buf, 'PNG')
        base64_data = ba.toBase64().data()

        if displayColCtr == 0:
            s += "<tr>"

        im = 'data:image/png;base64,' + base64_data

        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        #label = 'Range: [%.1f, %.1f]<br>Dims: [%d, %d]*%.2f' % (vmin, vmax, data.shape[0], data.shape[1], zoom)
        label = 'Range: [%.1f, %.1f]<br>Dims: [%d, %d]*%.2f' % (vmin, vmax, data.shape[0], data.shape[1], currZoom)
        #s += """<th><center><img src="%s"/>%s</center></th>"""%(im,label)
        #if title != "":
        #    title += "<br>"
        s += """<th><center>%s<img src="%s"/>%s</center></th>"""%(title,im,label)
        displayColCtr += 1
        if displayColCtr >= ncols:
            displayColCtr = 0
            s += "</tr>"

    s += "</table>"
    display( HTML ( s ) )

def css_styling():
    from IPython.core.display import HTML
    styles = open("custom.css", "r").read() #or edit path to custom.css
    return HTML(styles)

def draw_text( data, text, color = 255, pos = 'lr' ):
    from PIL.Image import fromarray
    from PIL.ImageDraw import Draw
    from PIL import ImageFont
    from numpy import asarray

    font = ImageFont.load_default()

    image = fromarray( data )
    draw = Draw( image )
    w, h = draw.textsize( text, font = font )

    position = {
        'ul': lambda iw, ih, tw, th: ( 2, 0 ),
        'ur': lambda iw, ih, tw, th: ( iw - tw - 2, 0 ),
        'll': lambda iw, ih, tw, th: ( 2, ih - th ),
        'lr': lambda iw, ih, tw, th: ( iw - tw - 2, ih - th ),
    }

    pos = position[ pos ]( data.shape[ 1 ], data.shape[ 0 ], w, h )

    draw.text( pos, text, fill = color, font = font )
    del draw

    return asarray( image )


def nbvideo( data, vmin = None, vmax = None, vsym = False, fps = 1, loop = False, theora = 8, h264 = 18, vp8 = None, saveas = None, counter = None, counter_color = 255 ):
    '''
    Display raw data as a notebook inline video (using HTML5).

    Parameters:
    data: array-like object, three or four dimensions. First dimension
          is time. If four dimensional, second or last axis must have
          length 3 or 4 and will be interpreted as color (RGB or RGBA).
    vmin, vmax, vsym: refer to rerange()
    fps: Number of frames (pictures) per second that will be rendered. [ 1 ]
    loop: When finished playing, start playing from the beginning
          immediately [ False ]
    theora: Video quality setting for Ogg Theora codec, integer from
            0 (worst) to 10 (best). 8 is a good starting point. To disable
            Theora, set to None. [ 8 ]
    h264: Video quality setting for H.264/AVC (MP4) codec, integer from
          0 (best) to 51 (worst). 18 is a good starting point. To disable
          H.264/AVC, set to None. [ 18 ]
    vp8: Video quality setting for VP8 (WebM) codec, integer from
         4 (best) to 63 (worst). 12 is a good starting point. To disable
         VP8, set to None. [ None ]
    saveas: Save video file(s) to disk (optional). Proper file name extensions
            will be appended to the pathname given. [ None ]

    Notes:
    Getting the quality setting right may require some experimentation.
    Generally, the better the quality of the video, the greater the size of
    the encoded data will be. This will increase the file size of the IPython
    notebook. The default values provide a quality such that compression
    artifacts should be barely noticeable. Some browsers only support certain
    codecs (http://en.wikipedia.org/wiki/HTML5_video#Browser_support). With
    the default settings, the video should be playable in all current browsers.
    In order to save space (and/or encoding time), you may want to disable
    all codecs except one that your browser supports, by setting their quality
    parameters to None.
    '''
    from IPython.display import display, HTML
    from numpy import asarray
    from subprocess import Popen, PIPE

    data = rerange( data, vmin, vmax, vsym )
    data = data.squeeze()
    # try to be smart
    if 3 <= data.shape[ 1 ] <= 4:
        data = data.transpose( ( 0, 2, 3, 1 ) )

    if counter is not None:
        data = asarray( [ draw_text( d, str( i ), color = counter_color, pos = counter ) for i, d in enumerate( data ) ] )

    cmdstring  = ( 'ffmpeg',
        '-f', 'rawvideo',
        '-r', '{}'.format( int( fps ) ),
        '-s', '{}x{}'.format( data.shape[ 2 ], data.shape[ 1 ] ),
        '-pix_fmt', 'gray' if data.ndim == 3 else { 3: 'rgb24', 4: 'rgba' }[ data.shape[ -1 ] ],
        '-i', '-',
    )
    data = data.tostring()

    if loop:
        html = '<video controls loop>'
    else:
        html = '<video controls>'

    if theora is not None:
        cmdstring_add = ( '-f', 'ogg', '-codec', 'libtheora',
            '-pix_fmt', 'yuv420p', '-flags', 'qscale', '-q:v', str( int( theora ) ), '-' )
        p = Popen( cmdstring + cmdstring_add, stdin = PIPE, stdout = PIPE, shell = False )
        s = p.communicate( data )[ 0 ]
        if saveas is not None:
            open( saveas + '.ogg', 'wb' ).write( s )
        html += '<source src="data:video/ogg;base64,{}" type="video/ogg" />'.format( s.encode( 'base64' ) )

    if vp8 is not None:
        # set a very high maximum bitrate (-b) to let the bitstream size be determined by
        # video quality only
        cmdstring_add = ( '-f', 'webm', '-codec', 'libvpx',
            '-pix_fmt', 'yuv420p', '-crf', str( int( vp8 ) ), '-b:v', '100M', '-' )
        p = Popen( cmdstring + cmdstring_add, stdin = PIPE, stdout = PIPE, shell = False )
        s = p.communicate( data )[ 0 ]
        if saveas is not None:
            open( saveas + '.mkv', 'wb' ).write( s )
        html += '<source src="data:video/webm;base64,{}" type="video/webm" />'.format( s.encode( 'base64' ) )

    if h264 is not None:
        # mp4 format does not support non-seekable output -> create temp file if necessary
        from tempfile import mkstemp
        from os import unlink
        if saveas is not None:
            filename = saveas + '.mp4'
        else:
            filename = mkstemp()[ 1 ]
        try:
            cmdstring_add = ( '-f', 'mp4', '-codec', 'libx264',
                '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-qp', str( int( h264 ) ), '-y', filename )
            p = Popen( cmdstring + cmdstring_add, stdin = PIPE, shell = False )
            p.communicate( data )
            s = open( filename ).read()
        finally:
            if saveas is None:
                unlink( filename )
        html += '<source src="data:video/mp4;base64,{}" type="video/mp4" />'.format( s.encode( 'base64' ) )

    html += 'Your browser does not support the video tag.</video>'
    display( HTML( html ) )


def quilt( data, aspect = 1, max_x = None, space = 0, fill = 0, color_axis = None ):
    from numpy import ceil, inf, abs, empty, asarray, rollaxis
    data = asarray( data )
    if color_axis is None:
        data = data.reshape( ( -1, ) + data.shape[ -2: ] )
    else:
        # roll color axis to last one
        data = rollaxis( data, color_axis, data.ndim )
        data = data.reshape( ( -1, ) + data.shape[ -3: ] )
    best_cost = inf
    best_shape = None
    data_aspect = float( data.shape[ 2 ] ) / data.shape[ 1 ]
    if max_x is None:
        max_x = data.shape[ 0 ]
    else:
        max_x = ( max_x + space ) // ( data.shape[ 2 ] + space )
    for x in range( 1, max_x + 1 ):
        y = ceil( float( data.shape[ 0 ] ) / x )
        error = abs( data_aspect * x / y - aspect )
        if data.shape[ 0 ] % x > 0:
            error += float( x - data.shape[ 0 ] % x ) / x
        if error <= best_cost:
            best_cost = error
            best_shape = int( y ), int( x )
    out = empty( ( best_shape[ 0 ] * data.shape[ 1 ] + ( best_shape[ 0 ] - 1 ) * space,
                   best_shape[ 1 ] * data.shape[ 2 ] + ( best_shape[ 1 ] - 1 ) * space )
                 + data.shape[ 3: ], dtype = data.dtype )
    out[ :, : ] = fill
    for i in range( data.shape[ 0 ] ):
        y = i // best_shape[ 1 ]
        x = i % best_shape[ 1 ]
        out[ ( y * ( data.shape[ 1 ] + space ) ):( y * ( data.shape[ 1 ] + space ) + data.shape[ 1 ] ),
             ( x * ( data.shape[ 2 ] + space ) ):( x * ( data.shape[ 2 ] + space ) + data.shape[ 2 ] ) ] = data[ i ]
    return out


def dimwrap( data, axis = 0, aspect = 1, fill = 0 ):
    from numpy import ceil, inf, abs, empty, asarray, rollaxis
    data = asarray( data )
    best_cost = inf
    best_shape = None
    for x in range( 1, data.shape[ axis ] + 1 ):
        y = ceil( float( data.shape[ axis ] ) / x )
        error = abs( x / y - aspect )
        if data.shape[ axis ] % x > 0:
            error += float( x - data.shape[ axis ] % x ) / x
        if error <= best_cost:
            best_cost = error
            best_shape = int( y ), int( x )
    flatshape = data.shape[ :axis ] + ( best_shape[ 0 ] * best_shape[ 1 ], ) + data.shape[ ( axis + 1 ): ]
    newshape = data.shape[ :axis ] + best_shape + data.shape[ ( axis + 1 ): ]
    out = empty( flatshape, dtype = data.dtype )
    out[ tuple( slice( s ) for s in data.shape ) ] = data
    slices = tuple( [ slice( s ) for s in data.shape[ :axis ] ] +
                    [ slice( data.shape[ axis ], None ) ] +
                    [ slice( s ) for s in data.shape[ ( axis + 1 ): ] ] )
    out[ slices ].fill( fill )
    return out.reshape( newshape )
