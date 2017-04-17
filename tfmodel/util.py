# -*- coding: utf-8 -*-

import os
import tarfile
from six.moves import urllib


def maybe_download_and_extract(dest_directory, data_url):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(data_url, filepath)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
