#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Serves the singing horse demo via bottle.py and bjoern.

Author: Jan Schlüter
"""

import sys
import os
import base64
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import Image

import bottle
import numpy as np

import spects
#import predict  # for some reason, the app does not start up if we do this.
import plot

#bottle.TEMPLATE_PATH = ['.']
app = bottle.Bottle()

def load_config():
    here = os.path.dirname(__file__)
    config = dict()
    with open(os.path.join(here, 'config.ini'), 'rb') as f:
        for l in f:
            l = l.rstrip().split('=', 1)
            if len(l) == 2:
                config[l[0]] = l[1]
    return config

CONFIG = load_config()

@app.route('/', method='GET')
def start():
    """Serves the index.html page.

    Accepts(GET):
      No parameters.

    Returns:
      The index page, in HTML.
    """
    return bottle.static_file('index.html', root='.')

@app.route('/render', method='POST')
def render():
    """Computes and returns the audio and prediction curve for a given image.

    Accepts(POST):
      img: The drawing to add to the spectrogram, as a data URL in PNG format.

    Returns:
      the prediction curve image, as a data URL
      a CR LF sequence
      the resynthesized audio, as a data URL
    """
    # Obtain and decode given image data
    img = bottle.request.body.read()
    if not img:
        bottle.response = 500
        return 'Error: Missing img.'
    if not img.startswith('data:image/png;base64,'):
        bottle.response = 500
        return 'Error: Not a PNG data URL.'
    try:
        u = urlopen(img)
        img = u.read()
    except Exception:
        bottle.response = 500
        return 'Error: Could not decode data URL.'
    finally:
        u.close()
    try:
        img = StringIO(img)
        img = Image.open(img)
    except Exception:
        bottle.response = 500
        return 'Error: Could not load given data as image.'
    # Convert to numpy array, collapse channels if any
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., -1]  # use last channel only (alpha, or blue, or gray)
    img = img[::-1].T.astype(np.float32) / 255  # transpose and convert range
    # Draw image with 70% opacity on top of spectrogram
    img *= .7
    spect = np.load(os.path.join(os.path.dirname(__file__), 'static', 'original.npy'))
    spect = (spect - spects.VMIN) * (1 - img) + (spects.VMAX - spects.VMIN) * img
    spect += spects.VMIN
    # Resynthesize audio
    samples = spects.undo_melspect(spect)
    try:
        audio = spects.write_ffmpeg(samples)
    except OSError:
        audio = spects.write_ffmpeg(samples, cmd='avconv')
    # Compute network predictions
    import predict
    modeldir = os.path.join(os.path.dirname(__file__), CONFIG.get('modeldir'))
    preds = predict.predict(
            spect, os.path.join(modeldir, 'model.h5'),
            os.path.join(modeldir, 'std_mean.h5'))
    # Plot network predictions
    curve = plot.pred_curve(preds)
    return ("data:image/png;base64," + base64.b64encode(curve) + "\r\n" +
            "data:audio/mpeg;base64," + base64.b64encode(audio))


# Expose to mod_wsgi (we just need a global object called 'application'):
application = app

# Run as an internal server when this script is started directly:
def main():
    if len(sys.argv) > 1:
        print "Serves the demo. Needs bottle.py to run. Will serve via bjoern"
        print "if installed, otherwise via wsgi_ref. Reads its configuration "
        print "from config.ini."
        return

    # load configuration
    port = int(CONFIG.get('port', 9090))
    staticdir = os.path.join(os.path.dirname(__file__), 'static')

    # start web server
    print "Starting web server on localhost:%d..." % port
    app.route('/static/:path#.+#', callback=lambda path:
            bottle.static_file(path, root=staticdir))
    try:
        import bjoern
    except ImportError:
        bjoern = None
    bottle.run(app, host='localhost', port=port,
            server='bjoern' if bjoern else 'wsgiref')

if __name__=="__main__":
    main()

