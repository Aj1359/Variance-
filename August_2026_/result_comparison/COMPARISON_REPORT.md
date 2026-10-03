# Fair Influence Maximization: 34-Algorithm Comprehensive Benchmarking Report

This report provides a comparative study of 34 algorithms spanning existing baselines, topology-only variants, propagation-aware selections, and custom designs.

## 1. Global Performance Rankings

| Rank | Algorithm | Average Individual Fairness (Min P) [Higher is Fairer] | Average Spreading Power (Mean P) | Average Variance | Average Execution Time (s) |
|---|---|---|---|---|---|
| 1 | **PageRank Lookahead (ppr lookahead)** | 0.054400 | 0.6860 | 0.074447 | 5.578s |
| 2 | **K-Core Hybrid** | 0.052800 | 0.6899 | 0.073139 | 2.318s |
| 3 | **Hybrid Degree (avg degree)** | 0.048000 | 0.6912 | 0.072999 | 3.022s |
| 4 | **PageRank (ppr)** | 0.046400 | 0.6852 | 0.071353 | 3.132s |
| 5 | **Myopic BFS** | 0.044800 | 0.5688 | 0.038579 | 0.526s |
| 6 | **Naive Myopic** | 0.044800 | 0.6871 | 0.074026 | 0.168s |
| 7 | **ComponentFirst** | 0.041600 | 0.6889 | 0.073986 | 0.162s |
| 8 | **Gonzalez** | 0.041600 | 0.6876 | 0.074578 | 0.259s |
| 9 | **Concave Hybrid (concave)** | 0.041600 | 0.6855 | 0.072669 | 21.968s |
| 10 | **Myopic** | 0.040000 | 0.6918 | 0.072693 | 2.257s |
| 11 | **DegreeGonzalez** | 0.040000 | 0.6864 | 0.074899 | 0.240s |
| 12 | **HarmonicSpread** | 0.040000 | 0.4338 | 0.026106 | 0.248s |
| 13 | **MinDegree_hc** | 0.036800 | 0.5712 | 0.041845 | 0.187s |
| 14 | **PageRank Topo V1** | 0.035200 | 0.6868 | 0.073679 | 1.876s |
| 15 | **Naive Myopic PPR** | 0.035200 | 0.6683 | 0.073874 | 0.189s |
| 16 | **LeastCentral** | 0.032000 | 0.5404 | 0.038861 | 0.071s |
| 17 | **Naive Myopic BFS** | 0.030400 | 0.5961 | 0.052409 | 0.102s |
| 18 | **KCoreFrontier** | 0.027200 | 0.4750 | 0.027875 | 0.256s |
| 19 | **Prop-Aware Additive** | 0.025600 | 0.6876 | 0.076973 | 16.429s |
| 20 | **Myopic PPR** | 0.025600 | 0.6705 | 0.071128 | 2.097s |
| 21 | **Prop-Aware Multiplicative** | 0.024000 | 0.6894 | 0.078326 | 17.960s |
| 22 | **MinDegree_nd** | 0.024000 | 0.6793 | 0.075350 | 0.240s |
| 23 | **NeighborPPRBridge** | 0.022400 | 0.6839 | 0.079574 | 1.729s |
| 24 | **PageRank Topo V2** | 0.019200 | 0.6783 | 0.079871 | 1.900s |
| 25 | **TIM+** | 0.017600 | 0.6857 | 0.077563 | 1.493s |
| 26 | **BetweennessGateway** | 0.017600 | 0.6804 | 0.080571 | 0.309s |
| 27 | **EccentricitySpread** | 0.017600 | 0.5984 | 0.049563 | 0.301s |
| 28 | **Random** | 0.014400 | 0.6786 | 0.080034 | 0.090s |
| 29 | **PPR-Balance** | 0.014400 | 0.6759 | 0.080817 | 1.852s |
| 30 | **MinDegree_hcn** | 0.014400 | 0.6450 | 0.073396 | 0.275s |
| 31 | **EgoDensityBalance** | 0.012800 | 0.6784 | 0.080583 | 0.787s |
| 32 | **DegreeMedianSpread** | 0.011200 | 0.6781 | 0.080636 | 0.330s |
| 33 | **MinDegree_ndn** | 0.009600 | 0.6803 | 0.080349 | 0.267s |
| 34 | **LeastCentral_n** | 0.008000 | 0.6145 | 0.070528 | 0.143s |

## 2. Average Individual Fairness (Min P) per Network

| Algorithm | EU | Facebook | Irvine | ca-GrQc | ca-HepTh |
|---| --- | --- | --- | --- | --- |
| **PageRank Lookahead (ppr lookahead)** | 0.088000 | 0.128000 | 0.056000 | 0.000000 | 0.000000 |
| **K-Core Hybrid** | 0.072000 | 0.120000 | 0.072000 | 0.000000 | 0.000000 |
| **Hybrid Degree (avg degree)** | 0.080000 | 0.112000 | 0.048000 | 0.000000 | 0.000000 |
| **PageRank (ppr)** | 0.064000 | 0.128000 | 0.040000 | 0.000000 | 0.000000 |
| **Myopic BFS** | 0.088000 | 0.088000 | 0.048000 | 0.000000 | 0.000000 |
| **Naive Myopic** | 0.080000 | 0.104000 | 0.040000 | 0.000000 | 0.000000 |
| **ComponentFirst** | 0.072000 | 0.088000 | 0.048000 | 0.000000 | 0.000000 |
| **Gonzalez** | 0.064000 | 0.096000 | 0.048000 | 0.000000 | 0.000000 |
| **Concave Hybrid (concave)** | 0.056000 | 0.096000 | 0.056000 | 0.000000 | 0.000000 |
| **Myopic** | 0.040000 | 0.088000 | 0.072000 | 0.000000 | 0.000000 |
| **DegreeGonzalez** | 0.064000 | 0.080000 | 0.056000 | 0.000000 | 0.000000 |
| **HarmonicSpread** | 0.048000 | 0.080000 | 0.072000 | 0.000000 | 0.000000 |
| **MinDegree_hc** | 0.048000 | 0.104000 | 0.032000 | 0.000000 | 0.000000 |
| **PageRank Topo V1** | 0.080000 | 0.064000 | 0.032000 | 0.000000 | 0.000000 |
| **Naive Myopic PPR** | 0.080000 | 0.096000 | 0.000000 | 0.000000 | 0.000000 |
| **LeastCentral** | 0.096000 | 0.040000 | 0.024000 | 0.000000 | 0.000000 |
| **Naive Myopic BFS** | 0.056000 | 0.080000 | 0.016000 | 0.000000 | 0.000000 |
| **KCoreFrontier** | 0.040000 | 0.048000 | 0.048000 | 0.000000 | 0.000000 |
| **Prop-Aware Additive** | 0.000000 | 0.096000 | 0.032000 | 0.000000 | 0.000000 |
| **Myopic PPR** | 0.064000 | 0.064000 | 0.000000 | 0.000000 | 0.000000 |
| **Prop-Aware Multiplicative** | 0.000000 | 0.120000 | 0.000000 | 0.000000 | 0.000000 |
| **MinDegree_nd** | 0.072000 | 0.048000 | 0.000000 | 0.000000 | 0.000000 |
| **NeighborPPRBridge** | 0.000000 | 0.112000 | 0.000000 | 0.000000 | 0.000000 |
| **PageRank Topo V2** | 0.000000 | 0.096000 | 0.000000 | 0.000000 | 0.000000 |
| **TIM+** | 0.000000 | 0.088000 | 0.000000 | 0.000000 | 0.000000 |
| **BetweennessGateway** | 0.000000 | 0.088000 | 0.000000 | 0.000000 | 0.000000 |
| **EccentricitySpread** | 0.000000 | 0.088000 | 0.000000 | 0.000000 | 0.000000 |
| **Random** | 0.000000 | 0.072000 | 0.000000 | 0.000000 | 0.000000 |
| **PPR-Balance** | 0.000000 | 0.072000 | 0.000000 | 0.000000 | 0.000000 |
| **MinDegree_hcn** | 0.000000 | 0.072000 | 0.000000 | 0.000000 | 0.000000 |
| **EgoDensityBalance** | 0.000000 | 0.064000 | 0.000000 | 0.000000 | 0.000000 |
| **DegreeMedianSpread** | 0.000000 | 0.056000 | 0.000000 | 0.000000 | 0.000000 |
| **MinDegree_ndn** | 0.000000 | 0.048000 | 0.000000 | 0.000000 | 0.000000 |
| **LeastCentral_n** | 0.000000 | 0.040000 | 0.000000 | 0.000000 | 0.000000 |

## 3. Average Spreading Power (Mean P) per Network

| Algorithm | EU | Facebook | Irvine | ca-GrQc | ca-HepTh |
|---| --- | --- | --- | --- | --- |
| **PageRank Lookahead (ppr lookahead)** | 0.8890 | 0.9405 | 0.7425 | 0.3913 | 0.4665 |
| **K-Core Hybrid** | 0.8884 | 0.9414 | 0.7428 | 0.4075 | 0.4695 |
| **Hybrid Degree (avg degree)** | 0.8901 | 0.9433 | 0.7430 | 0.4063 | 0.4732 |
| **PageRank (ppr)** | 0.8897 | 0.9429 | 0.7418 | 0.3810 | 0.4704 |
| **Myopic BFS** | 0.8905 | 0.9409 | 0.7146 | 0.1064 | 0.1918 |
| **Naive Myopic** | 0.8895 | 0.9432 | 0.7415 | 0.3933 | 0.4679 |
| **ComponentFirst** | 0.8878 | 0.9409 | 0.7405 | 0.4045 | 0.4709 |
| **Gonzalez** | 0.8888 | 0.9421 | 0.7430 | 0.3970 | 0.4673 |
| **Concave Hybrid (concave)** | 0.8906 | 0.9414 | 0.7424 | 0.3850 | 0.4680 |
| **Myopic** | 0.8904 | 0.9426 | 0.7435 | 0.4079 | 0.4745 |
| **DegreeGonzalez** | 0.8891 | 0.9394 | 0.7422 | 0.3943 | 0.4667 |
| **HarmonicSpread** | 0.5652 | 0.9188 | 0.6737 | 0.0075 | 0.0039 |
| **MinDegree_hc** | 0.8888 | 0.9430 | 0.7024 | 0.1639 | 0.1577 |
| **PageRank Topo V1** | 0.8882 | 0.9408 | 0.7384 | 0.3961 | 0.4705 |
| **Naive Myopic PPR** | 0.8199 | 0.9376 | 0.7358 | 0.3852 | 0.4629 |
| **LeastCentral** | 0.7379 | 0.9160 | 0.6987 | 0.2176 | 0.1318 |
| **Naive Myopic BFS** | 0.8729 | 0.8655 | 0.6562 | 0.2740 | 0.3116 |
| **KCoreFrontier** | 0.6476 | 0.9406 | 0.6903 | 0.0372 | 0.0594 |
| **Prop-Aware Additive** | 0.8658 | 0.9428 | 0.7428 | 0.4115 | 0.4753 |
| **Myopic PPR** | 0.8629 | 0.9398 | 0.7240 | 0.3712 | 0.4547 |
| **Prop-Aware Multiplicative** | 0.8747 | 0.9453 | 0.7362 | 0.4145 | 0.4763 |
| **MinDegree_nd** | 0.8750 | 0.9360 | 0.7339 | 0.3872 | 0.4646 |
| **NeighborPPRBridge** | 0.8649 | 0.9411 | 0.7354 | 0.4091 | 0.4687 |
| **PageRank Topo V2** | 0.8637 | 0.9405 | 0.7276 | 0.3939 | 0.4658 |
| **TIM+** | 0.8818 | 0.9423 | 0.7373 | 0.3998 | 0.4672 |
| **BetweennessGateway** | 0.8639 | 0.9389 | 0.7295 | 0.4032 | 0.4666 |
| **EccentricitySpread** | 0.8808 | 0.9414 | 0.7363 | 0.2089 | 0.2247 |
| **Random** | 0.8674 | 0.9327 | 0.7331 | 0.3942 | 0.4656 |
| **PPR-Balance** | 0.8641 | 0.9271 | 0.7302 | 0.3950 | 0.4629 |
| **MinDegree_hcn** | 0.8690 | 0.9382 | 0.5557 | 0.3967 | 0.4654 |
| **EgoDensityBalance** | 0.8650 | 0.9325 | 0.7299 | 0.3978 | 0.4666 |
| **DegreeMedianSpread** | 0.8627 | 0.9350 | 0.7330 | 0.3944 | 0.4654 |
| **MinDegree_ndn** | 0.8660 | 0.9391 | 0.7351 | 0.3964 | 0.4650 |
| **LeastCentral_n** | 0.8669 | 0.9227 | 0.4215 | 0.3952 | 0.4660 |

## 4. Folder Structure Reference

- **`existing_14/`**: Contains the 14 baseline algorithms (Random, Myopic, Naive Myopic, Gonzalez + 10 heuristics from Windham's paper).
- **`mine_algorithm/`**: Contains our 12 custom topology-only algorithms (focusing on variance minimization and structural coverage).
- **`new_two_algos/`**: Contains the 2 propagation-aware algorithms (Additive & Multiplicative).
- **`our_own/`**: PPR, PPR Lookahead, Concave Hybrid, Average Degree, and K-Core Hybrid algorithms.
