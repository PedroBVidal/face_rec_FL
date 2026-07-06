import mxnet as mx
import os

data_path = "/work/pedro.vidal/dcface_rec/"
rec_path = os.path.join(data_path, 'train.rec')
idx_path = os.path.join(data_path, 'train.idx')

imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
print(f"Header at idx 0: flag={header.flag}, label={header.label}")

id_start = int(header.label[0])
id_end = int(header.label[1])
print(f"Identity index range: {id_start} to {id_end - 1}")
print(f"Total identities: {id_end - id_start}")

# Check first identity
s_id = imgrec.read_idx(id_start)
header_id, _ = mx.recordio.unpack(s_id)
print(f"Header for first identity (idx {id_start}): label={header_id.label}")
