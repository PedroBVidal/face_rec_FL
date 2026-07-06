import mxnet as mx
import os
import time

data_path = "/work/pedro.vidal/dcface_rec/"
rec_path = os.path.join(data_path, 'train.rec')
idx_path = os.path.join(data_path, 'train.idx')

start_time = time.time()
imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
keys = list(imgrec.keys)
print(f"Total keys: {len(keys)}")

import numbers
label_map = {}
for i, idx in enumerate(keys):
    s = imgrec.read_idx(idx)
    header, _ = mx.recordio.unpack(s)
    label = header.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    label = int(label)
    if label not in label_map:
        label_map[label] = []
    label_map[label].append(idx)

print(f"Total unique identities: {len(label_map)}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
