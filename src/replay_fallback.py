"""One bounded alternative to the archived replay; no original operator changes."""
import numpy as np


def replay_cells(values, trace, reconstruction, times, origins, fallback='hold'):
    x = np.asarray(values, dtype=float)
    s, o = np.asarray(times), np.asarray(origins)
    if s.shape != o.shape or np.any(s < 0) or np.any(s > o) or np.any(o >= len(x)):
        raise ValueError('invalid feature/origin geometry')
    if fallback not in ('hold', 'capped_slope'):
        raise ValueError('unknown fallback')
    ready = reconstruction.availability[s] <= o
    pos = np.minimum(np.searchsorted(trace.event_indices, s, side='right') - 1,
                     np.searchsorted(trace.commit_indices, o, side='right') - 1)
    if np.any(pos < 0):
        raise ValueError('no eligible past source')
    last = trace.event_indices[pos]
    prev_pos = np.maximum(0, pos-1)
    previous = trace.event_indices[prev_pos]
    result = x[last].copy()
    if fallback == 'capped_slope':
        gap = last - previous
        fraction = np.minimum(s-last, gap) / np.maximum(gap, 1)
        result += (x[last]-x[previous]) * fraction
    result = np.where(ready, reconstruction.values[s], result)
    used_commit = np.where(ready, reconstruction.availability[s], trace.commit_indices[pos])
    if np.any(used_commit > o) or np.any(last > s):
        raise AssertionError('replay source after origin or fallback after feature time')
    return result
