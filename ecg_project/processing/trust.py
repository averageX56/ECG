"""Foreground probability bands; background is never an uncertain ECG wave."""
import numpy as np


def decode_probability_bands(probs, signal, fs, original_length, low=.3, high=.5,
                             smoothing_ms=20., max_extension_ms=40.):
    """250 Hz probabilities -> confident core plus attached foreground envelope.

    Each envelope is a contiguous component whose winning class is P/QRS/T
    and whose probability is >= low. A >= high core (at least 5 samples)
    is required. Components separated by background are never joined.
    """
    if not 0 <= low < high <= 1:
        raise ValueError('Require 0 <= low < high <= 1')
    if not np.isfinite(smoothing_ms) or not np.isfinite(max_extension_ms) or min(smoothing_ms, max_extension_ms) < 0:
        raise ValueError('Nonnegative finite smoothing/extension required')
    # Smooth probabilities, not a binary mask. Tiny confidence flicker no
    # longer creates a piano of alternating foreground/background samples.
    size = max(1, round(smoothing_ms*250/1000))
    size += (size % 2 == 0)
    probs = np.asarray(probs, dtype=np.float64)
    if size > 1:
        probs = np.stack([np.convolve(np.pad(probs[:, c], (size//2, size//2), mode='edge'),
                                     np.ones(size)/size, mode='valid') for c in range(4)], axis=1)
    extension = round(max_extension_ms*250/1000)
    winner = probs.argmax(-1)
    waves = []
    baseline = np.median(signal)
    def components(mask):
        change = np.diff(np.r_[False, mask, False].astype(int))
        return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))
    def coordinate(i):
        return min(round(int(i)*fs/250), original_length-1)
    for cls, name in ((1, 'P'), (2, 'QRS'), (3, 'T')):
        for a, b in components((winner == cls) & (probs[:, cls] >= low)):
            cores = [(a+c, a+d) for c, d in components(probs[a:b, cls] >= high) if d-c >= 5]
            if not cores:
                continue
            # Multiple confident islands in one foreground component describe
            # one wave; their first/last samples delimit its confident extent.
            c, d = cores[0][0], cores[-1][1]
            a, b = max(a, c-extension), min(b, d+extension)
            peak = c + int(np.argmax(abs(signal[c:d]-baseline)))
            waves.append(dict(wave=name, onset=coordinate(c), offset=coordinate(d-1), peak=coordinate(peak),
                              onset_lower=coordinate(a), onset_upper=coordinate(c),
                              offset_lower=coordinate(d-1), offset_upper=coordinate(b-1),
                              confidence=float(probs[c:d, cls].mean()), source='probability_band'))
    return sorted(waves, key=lambda w: w['onset'])


def feature_bands(point, peaks, waves, fs):
    """Propagate onset/offset envelopes to QRS/PR/QT/T feature ranges."""
    point = np.asarray(point, np.float32).copy()
    point[point[:, 7] == 0, 7] = np.nan  # Undetected P is unknown, not proven absent.
    lower, upper = point.copy(), point.copy()
    confidence = np.full_like(point, np.nan)
    qs = [w for w in waves if w['wave'] == 'QRS']
    ps = [w for w in waves if w['wave'] == 'P']
    ts = [w for w in waves if w['wave'] == 'T']
    def duration(left, right):
        return (max(0, right['offset_lower']-left['onset_upper'])*1000/fs,
                max(0, right['offset_upper']-left['onset_lower'])*1000/fs)
    widths = np.asarray([duration(q, q) for q in qs])
    median_widths = np.median(widths, axis=0) if len(widths) else [np.nan, np.nan]
    for i, peak in enumerate(peaks):
        near = [q for q in qs if abs(q['peak']-peak) <= .15*fs]
        if not near:
            continue
        q = min(near, key=lambda w: abs(w['peak']-peak))
        lower[i, 5], upper[i, 5] = duration(q, q)
        confidence[i, 5] = q['confidence']
        if median_widths[0] > 0:
            lower[i, 10], upper[i, 10] = lower[i, 5]/median_widths[1], upper[i, 5]/median_widths[0]
            confidence[i, 10] = min(w['confidence'] for w in qs)
        else:
            lower[i, 10] = upper[i, 10] = np.nan
        prior = [p for p in ps if 0 < q['onset']-p['offset'] < .35*fs]
        if prior:
            p = max(prior, key=lambda w: w['offset'])
            lower[i, 6] = max(0, q['onset_lower']-p['onset_upper'])*1000/fs
            upper[i, 6] = max(0, q['onset_upper']-p['onset_lower'])*1000/fs
            confidence[i, 6] = confidence[i, 7] = min(p['confidence'], q['confidence'])
        following = [t for t in ts if 0 < t['onset']-q['offset'] < .6*fs]
        if following:
            t = min(following, key=lambda w: w['onset'])
            lower[i, 11], upper[i, 11] = duration(q, t)
            lower[i, 12], upper[i, 12] = duration(t, t)
            confidence[i, 11] = min(q['confidence'], t['confidence'])
            confidence[i, 12] = t['confidence']
    return point, np.concatenate([point, lower, upper, upper-lower, confidence], axis=1)
