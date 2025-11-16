# Min-CPS/Min-RCPS
---
Ordinal classification has been widely applied in many high-
stakes applications, e.g., medical imaging and diagnosis,
where reliable uncertainty quantification (UQ) is essential for
decision making. Conformal prediction (CP) is a general UQ
framework that provides statistically valid guarantees, which
is especially useful in practice. However, prior ordinal CP
methods mainly focus on heuristic algorithms or restrictively
require the underlying model to predict a unimodal distribu-
tion over ordinal labels. Consequently, they lack either a com-
prehensive understanding of the coverage optimality and pre-
dictive efficiency, or a model-agnostic and distribution-free
nature favored by CP methods. To this end, we fill this gap
by proposing the first ordinal-CP method that is both model-
agnostic and provably optimal. Specifically, we formulate
conformal ordinal classification as an expected minimum-
length covering problem over the underlying distribution. To
solve this problem, we develop a sliding-window algorithm
that is optimal each calibration example, with only a linear
time complexity in K, the number of label candidates. The lo-
cal optimality per instance further yields global optimality in
expectation. Moreover, we propose a length-regularized vari-
ant that shrinks prediction set size while preserving coverage.
Extensive experiments on three benchmark datasets from di-
verse domains have been conducted to demonstrate the signif-
icantly improved predictive efficiency of the proposed meth-
ods over baselines (by 18%↓ on average over three datasets).

## Running instructions

Please run the commands mentioned below to produce results:
