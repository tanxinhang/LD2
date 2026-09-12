"""Opt-in reserved-slot STOP transport with action-independent slot randomness."""
import numpy as np


class ReservedSlotTransport:
    """Own a dedicated radio; do not mix calls with its legacy RNG schedule.

    Each reserved slot advances the existing radio once, including silence.
    Slot-keyed randomness prevents packet draws changing future shadowing.
    This is a new simulation coupling, not legacy seeded-trace equivalence.
    """
    def __init__(self,radio,seed):
        self.radio=radio
        self.seed=int(seed)
        self.slot=0

    def send(self,positions,sender=None):
        self.radio.rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.slot]))
        self.slot+=1
        if sender is None:
            return self.radio.transmit({}, {},positions)
        return self.radio.transmit({sender:np.zeros(self.radio.message_dim)},
            {sender:0},positions,tx_powers_w={sender:.1},
            extra_payload_bits={sender:64},base_payload_dimensions={sender:0},
            suppress_message_payload={sender:True})

    def stop_control(self,positions,request_stop):
        packets,stats=self.send(positions,1 if request_stop else None)
        stopped=bool(request_stop and any(p.sender==1 and p.receiver==2 for p in packets))
        return stopped,stats
