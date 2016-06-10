import os
import sys

import numpy as np
from progressbar import ProgressBar

from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve


def download_file(fname, origin):
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, 'models')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)
    if os.path.exists(fpath):
        return fpath

    print('Downloading data from',  origin)

    global progbar
    progbar = None

    def dl_progress(count, block_size, total_size):
        global progbar
        if progbar is None:
            progbar = ProgressBar(max_value=total_size)
        elif count*block_size < total_size:
            progbar.update(count*block_size)
        else:
            progbar.finish()

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath, dl_progress)
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)
        raise e

    return fpath

def load_np_data(fname):
    dirname = os.path.dirname(__file__)
    datapath = os.path.join(dirname, '..', 'data', fname)
    data = np.load(datapath)
    return data
