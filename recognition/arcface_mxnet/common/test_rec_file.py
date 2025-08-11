import os, sys
import mxnet as mx
import numpy as np
from PIL import Image
import io

path_rec = '/home/pbqv20/datasets/datamxrec_dcface1M'

record = mx.recordio.MXIndexedRecordIO(os.path.join(path_rec,'train.idx'), os.path.join(path_rec,'train.rec'), 'r')

for idx in range(5):  # Try the first few records
    try:
        s = record.read_idx(idx)
        header, content = mx.recordio.unpack(s)
        print(f"Index: {idx}, Flag: {header.flag}, Label: {header.label}")
        if header.flag > 0:
            img = mx.image.imdecode(content).asnumpy()
            Image.fromarray(img).show()  # View the image
        else:
            print("Not an image record.")
    except Exception as e:
        print(f"Error at idx {idx}: {e}")
