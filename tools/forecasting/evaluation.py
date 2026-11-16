"""Pooled CPU forecasting metrics, using the verified occupancy evaluator."""
import numpy as np

from tools import evaluate_occupancy as occupancy


def statistics(forecast, target, target_in_bounds, target_known):
    arrays = [np.asarray(a) for a in (forecast, target, target_in_bounds, target_known)]
    if any(a.shape != arrays[0].shape for a in arrays) or target_in_bounds.dtype != bool or target_known.dtype != bool:
        raise ValueError('Forecast/target/masks shape or boolean dtype mismatch')
    # This mask has no dependence on confidence, prediction labels or correctness.
    valid = target_in_bounds & target_known
    if np.any(valid & (target == 255)) or np.any((~valid) & (target != 255)):
        raise ValueError('Target labels and target-only mask disagree')
    return occupancy.confusion_statistics(forecast, target)


def pooled(matrices):
    matrices = list(matrices)
    return occupancy.summarize(np.sum(matrices, axis=0, dtype=np.int64)) if matrices else None
