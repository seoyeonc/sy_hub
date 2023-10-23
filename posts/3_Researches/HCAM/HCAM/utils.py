import numpy as np
from fastai.vision.all import *


def label_func(f):
    if f[0].isupper():
        return 'cat' 
    else: 
        return 'dog' 


def load_data(fpath, nresize=512):
    path=Path(fpath)
    files=get_image_files(path)
    dls=ImageDataLoaders.from_name_func(path,files,label_func,item_tfms=Resize(nresize))
    return dls