"""Model/offline-log based admission, never instantaneous reverse-link CSI."""
import math
from scipy.stats import beta


def delivery_lower_bound(successes,trials,error=.05):
    if not isinstance(successes,int) or not isinstance(trials,int) or not 0<=successes<=trials or not 0<error<1:
        raise ValueError('valid binomial counts and error required')
    if trials==0 or successes==0: return 0.
    return float(beta.ppf(error,successes,trials-successes+1))


def admit_control(delivery_lower,control_bits,cancellable_bits):
    if not all(math.isfinite(x) for x in (delivery_lower,control_bits,cancellable_bits)):
        raise ValueError('finite values required')
    if not 0<=delivery_lower<=1 or control_bits<0 or cancellable_bits<0:
        raise ValueError('invalid bounds or costs')
    # Must bound P(delivered | request, information stratum), not blindly
    # substitute an unconditional success frequency after selective requests.
    return delivery_lower*cancellable_bits>control_bits
