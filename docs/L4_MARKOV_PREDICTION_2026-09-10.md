# L4 causal Markov prediction

The L4 plant and local belief both use the same linear-Gaussian CV transition
with acceleration standard deviation 1.5 m/s2. The continuous target state is
therefore already Markov. Introducing an unsupported discrete CT/CA regime
chain would change the model rather than improve inference. This experiment
instead applies the exact Chapman--Kolmogorov recursion and uses the predicted
one-second target mean in the existing distributed bistatic assignment.

The predictor is causal, consumes each viewer's existing local belief mean and
velocity, reflects the mean at the simulator boundary, and does not use future
measurements or target truth. Frozen profiles retain zero prediction frames.
The assignment hold remains 20 frames; power, communication, sensing budget
and candidate count are unchanged.

On four calibration seeds, H10 changed steady/weak3/worst from
0.9000/0.8287/0.8051 to 0.9092/0.8577/0.8460 and raised the minimum worst score
from 0.6250 to 0.6791. H20 improved the mean less and caused a 0.06 single-seed
regression. H40 reduced worst to 0.7650 and the minimum to 0.4588, showing that
long open-loop mean prediction is harmful.

Before confirmation, eight untouched non-test seeds were selected with fixed
RNG seed 20260910 across the stored easy/medium/hard geometry strata. The exact
seed list and acceptance rules are frozen in
`artifacts/research/l4_markov_prediction_h10_v1.json`. This is development
confirmation, not formal evidence; a larger bank is required for a narrow
confidence interval.

The preregistered development confirmation rejected the endpoint-mean H10
candidate. On the eight untouched seeds, baseline versus candidate
steady/weak3/worst was 0.94847/0.89385/0.87547 versus
0.94277/0.88661/0.86939. The candidate worst minimum was 0.72881 versus
0.75218, with four wins and four losses. SCORE, WEAK3 and TAIL therefore
failed; QoS, L0 and resource guards passed. The next hypothesis must integrate
the path distribution and belief covariance rather than tune another fixed
endpoint horizon.
