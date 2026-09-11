"""Owner-local, windowed energy evidence from actual delivery events."""
import numpy as np


class DeliveredEnergyHistory:
    def __init__(self, window_blocks):
        if not isinstance(window_blocks, int) or window_blocks < 1:
            raise ValueError('window_blocks must be a positive integer')
        self.window_blocks = window_blocks
        self.entries = {}
        self.now = -1

    def advance(self, frame):
        if frame < self.now:
            raise ValueError('history clock cannot move backwards')
        self.now = frame
        self.entries = {key:value for key,value in self.entries.items()
                        if 0 <= frame-key[0] < self.window_blocks}

    def accept(self, evidence_id, energy, *, available):
        if not available:
            return False
        frame, source, target = evidence_id
        if frame > self.now:
            raise ValueError('future evidence rejected')
        if self.now-frame >= self.window_blocks:
            return False
        value=np.asarray(energy,dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value<0):
            raise ValueError('energy must be finite and nonnegative')
        if evidence_id in self.entries:
            if not np.array_equal(value,self.entries[evidence_id]):
                raise ValueError('conflicting duplicate evidence')
            return False
        if self.entries and value.shape != next(iter(self.entries.values())).shape:
            raise ValueError('inconsistent trial dimensions')
        self.entries[evidence_id]=value.copy()
        return True

    def total(self):
        if not self.entries:
            return 0, 0.
        return len(self.entries),np.sum(list(self.entries.values()),axis=0)
