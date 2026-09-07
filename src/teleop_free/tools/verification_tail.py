"""A contiguous episode prefix; never splice away intermediate pauses."""
import numpy as np


def training_prefix_length(seqs, num_rows, row):
    marker = row.get('execution', {}).get('metrics', {}).get('training_tail')
    if not row.get('ok') or marker is None:
        raise ValueError('verification trimming requires a successful episode with training_tail')
    seqs = np.asarray(seqs, dtype=np.int64)
    if len(seqs) != num_rows or len(seqs) < 2 or np.any(np.diff(seqs) < 0):
        raise ValueError('invalid exporter sequence join for verification trimming')
    end = marker['end_seq']
    if type(end) is not int or not seqs[0] < end < seqs[-1]:
        raise ValueError('training boundary outside recorded sequence range')
    return int(np.searchsorted(seqs, end, side='right'))
