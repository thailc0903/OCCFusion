"""Input-only persistence baseline; no dataset, future pose, GT or file access."""
from dataclasses import dataclass
import numpy as np

from tools.forecasting.protocol import HORIZONS_US


@dataclass(frozen=True)
class HistoryInputs:
    labels: np.ndarray
    valid: np.ndarray
    timestamps_us: tuple[int, ...]
    used_sensor_timestamps_us: tuple[tuple[int, ...], ...]
    anchor_timestamp_us: int
    input_cutoff_us: int

    def validate(self):
        if self.labels.ndim != 4 or self.labels.shape[0] != 3 or self.valid.shape != self.labels.shape or self.valid.dtype != bool:
            raise ValueError('Three aligned XYZ history grids and independent boolean valid masks required')
        if len(self.timestamps_us) != 3 or len(self.used_sensor_timestamps_us) != 3 or tuple(sorted(set(self.timestamps_us))) != self.timestamps_us:
            raise ValueError('History timestamps must be distinct and increasing')
        if self.timestamps_us[-1] != self.anchor_timestamp_us or self.input_cutoff_us != self.anchor_timestamp_us:
            raise ValueError('Anchor/cutoff must equal the final history LiDAR time')
        if any(not times or max(times) > self.input_cutoff_us for times in self.used_sensor_timestamps_us):
            raise ValueError('Noncausal history sensor after input cutoff')
        if self.labels.dtype.kind not in 'iu' or np.any((self.labels < 0) | ((self.labels > 16) & (self.labels != 255))):
            raise ValueError('Invalid semantic history labels')
        if not self.valid[-1].all() or np.any(self.labels[-1] == 255):
            raise ValueError('Persistence requires complete saved anchor prediction, cannot fill unknown as free')


def predict(history: HistoryInputs):
    """Never accepts targets/future poses/probability; history[-1] is the anchor."""
    history.validate()
    return np.repeat(history.labels[-1][None], len(HORIZONS_US), axis=0)
