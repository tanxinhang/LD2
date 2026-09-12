"""Offline admission check; never consume an evaluation stop rate online."""
def report_budget_check(full_bits,control_bits,cancellable_bits,stop_probability_lower,budget_bits):
    import math
    values=(full_bits,control_bits,cancellable_bits,stop_probability_lower,budget_bits)
    if not all(math.isfinite(x) and x>=0 for x in values) or stop_probability_lower>1:
        raise ValueError('invalid nonnegative budget inputs')
    if cancellable_bits>full_bits: raise ValueError('cannot cancel more than full schedule')
    upper=full_bits+control_bits-cancellable_bits*stop_probability_lower
    return dict(expected_bits_upper=upper,admissible=upper<=budget_bits,
        required_stop_probability=(max(0.,(full_bits+control_bits-budget_bits)/cancellable_bits)
            if cancellable_bits else (0. if full_bits+control_bits<=budget_bits else None)))
