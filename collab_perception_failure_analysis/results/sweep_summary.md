# Sweep summary (mean ± std over seeds)

Floor (ego-only) AP@0.7 = 0.575, margin ±0.02

## attfuse

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| bandwidth | L0 (16) | 0.814 ± 0.000 | 0.889 | 0.900 | 1.59 | above_floor |
| bandwidth | L1 (8) | 0.814 ± 0.001 | 0.889 | 0.900 | 1.59 | above_floor |
| bandwidth | L2 (4) | 0.810 ± 0.000 | 0.885 | 0.897 | 1.59 | above_floor |
| bandwidth | L3 (2) | 0.735 ± 0.000 | 0.863 | 0.834 | 1.59 | above_floor |
| bandwidth | L4 (1) | 0.556 ± 0.000 | 0.793 | 0.686 | 1.59 | at_floor |
| ghosts | L0 (1) | 0.805 ± 0.001 | 0.877 | 0.899 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.794 ± 0.001 | 0.866 | 0.898 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.773 ± 0.002 | 0.843 | 0.896 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.741 ± 0.003 | 0.806 | 0.892 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.685 ± 0.002 | 0.740 | 0.884 | 1.59 | above_floor |
| latency | L0 (1) | 0.520 ± 0.000 | 0.712 | 0.714 | 1.59 | below_floor |
| latency | L1 (2) | 0.399 ± 0.000 | 0.626 | 0.620 | 1.59 | below_floor |
| latency | L2 (4) | 0.370 ± 0.000 | 0.562 | 0.629 | 1.59 | below_floor |
| latency | L3 (6) | 0.365 ± 0.000 | 0.538 | 0.644 | 1.59 | below_floor |
| latency | L4 (8) | 0.358 ± 0.000 | 0.530 | 0.641 | 1.59 | below_floor |
| latency | L5 (10) | 0.359 ± 0.000 | 0.525 | 0.647 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.797 ± 0.002 | 0.880 | 0.887 | 1.42 | above_floor |
| loss_burst | L1 (0.3) | 0.751 ± 0.010 | 0.856 | 0.858 | 1.10 | above_floor |
| loss_burst | L2 (0.5) | 0.710 ± 0.004 | 0.836 | 0.829 | 0.81 | above_floor |
| loss_burst | L3 (0.7) | 0.654 ± 0.006 | 0.809 | 0.787 | 0.49 | above_floor |
| loss_burst | L4 (0.9) | 0.627 ± 0.002 | 0.796 | 0.770 | 0.37 | above_floor |
| loss_iid | L0 (0.1) | 0.796 ± 0.002 | 0.880 | 0.888 | 1.43 | above_floor |
| loss_iid | L1 (0.3) | 0.756 ± 0.004 | 0.860 | 0.859 | 1.10 | above_floor |
| loss_iid | L2 (0.5) | 0.711 ± 0.002 | 0.838 | 0.831 | 0.82 | above_floor |
| loss_iid | L3 (0.7) | 0.655 ± 0.005 | 0.810 | 0.789 | 0.49 | above_floor |
| loss_iid | L4 (0.9) | 0.586 ± 0.004 | 0.772 | 0.736 | 0.15 | at_floor |
| pose | L0 (0.2) | 0.681 ± 0.006 | 0.817 | 0.817 | 1.59 | above_floor |
| pose | L1 (0.4) | 0.502 ± 0.010 | 0.708 | 0.689 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.394 ± 0.004 | 0.647 | 0.585 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.416 ± 0.002 | 0.690 | 0.583 | 1.55 | below_floor |
| pose | L4 (3.2) | 0.460 ± 0.009 | 0.724 | 0.614 | 1.33 | below_floor |
| stale | L0 (2) | 0.659 ± 0.000 | 0.803 | 0.808 | 1.59 | above_floor |
| stale | L1 (4) | 0.506 ± 0.000 | 0.703 | 0.710 | 1.59 | below_floor |
| stale | L2 (8) | 0.429 ± 0.000 | 0.616 | 0.674 | 1.59 | below_floor |
| stale | L3 (16) | 0.389 ± 0.000 | 0.565 | 0.659 | 1.58 | below_floor |
| stale | L4 (32) | 0.369 ± 0.000 | 0.543 | 0.646 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.767 ± 0.006 | 0.850 | 0.882 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.665 ± 0.007 | 0.770 | 0.834 | 1.59 | above_floor |
| swap | L2 (0.5) | 0.569 ± 0.015 | 0.690 | 0.777 | 1.59 | at_floor |
| swap | L3 (0.75) | 0.451 ± 0.011 | 0.606 | 0.693 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.349 ± 0.001 | 0.520 | 0.602 | 1.59 | below_floor |

## coalign

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| bandwidth | L0 (16) | 0.835 ± 0.001 | 0.881 | 0.920 | 1.59 | above_floor |
| bandwidth | L1 (8) | 0.835 ± 0.000 | 0.881 | 0.920 | 1.59 | above_floor |
| bandwidth | L2 (4) | 0.818 ± 0.000 | 0.855 | 0.919 | 1.59 | above_floor |
| bandwidth | L3 (2) | 0.577 ± 0.001 | 0.657 | 0.838 | 1.59 | at_floor |
| bandwidth | L4 (1) | 0.506 ± 0.000 | 0.716 | 0.679 | 1.59 | below_floor |
| ghosts | L0 (1) | 0.813 ± 0.001 | 0.854 | 0.919 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.790 ± 0.001 | 0.829 | 0.918 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.752 ± 0.003 | 0.783 | 0.915 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.689 ± 0.001 | 0.706 | 0.913 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.601 ± 0.002 | 0.602 | 0.905 | 1.59 | above_floor |
| latency | L0 (1) | 0.546 ± 0.000 | 0.713 | 0.745 | 1.59 | below_floor |
| latency | L1 (2) | 0.442 ± 0.000 | 0.639 | 0.662 | 1.59 | below_floor |
| latency | L2 (4) | 0.407 ± 0.000 | 0.563 | 0.685 | 1.59 | below_floor |
| latency | L3 (6) | 0.402 ± 0.000 | 0.534 | 0.713 | 1.59 | below_floor |
| latency | L4 (8) | 0.399 ± 0.000 | 0.523 | 0.721 | 1.59 | below_floor |
| latency | L5 (10) | 0.391 ± 0.000 | 0.514 | 0.719 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.821 ± 0.003 | 0.874 | 0.910 | 1.43 | above_floor |
| loss_burst | L1 (0.3) | 0.784 ± 0.007 | 0.857 | 0.886 | 1.13 | above_floor |
| loss_burst | L2 (0.5) | 0.739 ± 0.006 | 0.836 | 0.855 | 0.81 | above_floor |
| loss_burst | L3 (0.7) | 0.682 ± 0.007 | 0.808 | 0.817 | 0.48 | above_floor |
| loss_burst | L4 (0.9) | 0.655 ± 0.004 | 0.794 | 0.797 | 0.33 | above_floor |
| loss_iid | L0 (0.1) | 0.817 ± 0.003 | 0.873 | 0.909 | 1.43 | above_floor |
| loss_iid | L1 (0.3) | 0.785 ± 0.005 | 0.857 | 0.886 | 1.14 | above_floor |
| loss_iid | L2 (0.5) | 0.739 ± 0.002 | 0.837 | 0.857 | 0.82 | above_floor |
| loss_iid | L3 (0.7) | 0.680 ± 0.007 | 0.806 | 0.816 | 0.47 | above_floor |
| loss_iid | L4 (0.9) | 0.619 ± 0.008 | 0.775 | 0.771 | 0.15 | above_floor |
| pose | L0 (0.2) | 0.695 ± 0.007 | 0.807 | 0.836 | 1.59 | above_floor |
| pose | L1 (0.4) | 0.534 ± 0.005 | 0.706 | 0.726 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.454 ± 0.001 | 0.651 | 0.665 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.474 ± 0.004 | 0.665 | 0.681 | 1.55 | below_floor |
| pose | L4 (3.2) | 0.511 ± 0.004 | 0.695 | 0.705 | 1.32 | below_floor |
| stale | L0 (2) | 0.679 ± 0.000 | 0.796 | 0.832 | 1.59 | above_floor |
| stale | L1 (4) | 0.539 ± 0.000 | 0.704 | 0.745 | 1.59 | below_floor |
| stale | L2 (8) | 0.461 ± 0.000 | 0.611 | 0.721 | 1.59 | below_floor |
| stale | L3 (16) | 0.421 ± 0.000 | 0.555 | 0.719 | 1.58 | below_floor |
| stale | L4 (32) | 0.404 ± 0.000 | 0.537 | 0.713 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.782 ± 0.004 | 0.830 | 0.907 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.684 ± 0.006 | 0.748 | 0.873 | 1.59 | above_floor |
| swap | L2 (0.5) | 0.590 ± 0.017 | 0.661 | 0.831 | 1.59 | at_floor |
| swap | L3 (0.75) | 0.483 ± 0.015 | 0.572 | 0.765 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.385 ± 0.003 | 0.491 | 0.696 | 1.59 | below_floor |

## cobevt

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| bandwidth | L0 (16) | 0.858 ± 0.000 | 0.933 | 0.908 | 1.59 | above_floor |
| bandwidth | L1 (8) | 0.858 ± 0.000 | 0.932 | 0.908 | 1.59 | above_floor |
| bandwidth | L2 (4) | 0.763 ± 0.000 | 0.874 | 0.864 | 1.59 | above_floor |
| bandwidth | L3 (2) | 0.322 ± 0.001 | 0.484 | 0.589 | 1.59 | below_floor |
| bandwidth | L4 (1) | 0.452 ± 0.000 | 0.680 | 0.614 | 1.59 | below_floor |
| ghosts | L0 (1) | 0.841 ± 0.001 | 0.914 | 0.906 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.827 ± 0.002 | 0.896 | 0.906 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.800 ± 0.001 | 0.864 | 0.903 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.753 ± 0.001 | 0.807 | 0.898 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.679 ± 0.004 | 0.719 | 0.889 | 1.59 | above_floor |
| latency | L0 (1) | 0.447 ± 0.000 | 0.676 | 0.643 | 1.59 | below_floor |
| latency | L1 (2) | 0.246 ± 0.000 | 0.491 | 0.457 | 1.59 | below_floor |
| latency | L2 (4) | 0.251 ± 0.000 | 0.452 | 0.488 | 1.59 | below_floor |
| latency | L3 (6) | 0.271 ± 0.000 | 0.459 | 0.527 | 1.59 | below_floor |
| latency | L4 (8) | 0.272 ± 0.000 | 0.459 | 0.532 | 1.59 | below_floor |
| latency | L5 (10) | 0.275 ± 0.000 | 0.459 | 0.544 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.847 ± 0.003 | 0.931 | 0.897 | 1.44 | above_floor |
| loss_burst | L1 (0.3) | 0.816 ± 0.003 | 0.927 | 0.869 | 1.13 | above_floor |
| loss_burst | L2 (0.5) | 0.779 ± 0.003 | 0.923 | 0.833 | 0.79 | above_floor |
| loss_burst | L3 (0.7) | 0.736 ± 0.003 | 0.917 | 0.793 | 0.49 | above_floor |
| loss_burst | L4 (0.9) | 0.716 ± 0.006 | 0.912 | 0.775 | 0.35 | above_floor |
| loss_iid | L0 (0.1) | 0.844 ± 0.004 | 0.931 | 0.895 | 1.43 | above_floor |
| loss_iid | L1 (0.3) | 0.818 ± 0.005 | 0.929 | 0.871 | 1.11 | above_floor |
| loss_iid | L2 (0.5) | 0.783 ± 0.005 | 0.924 | 0.836 | 0.80 | above_floor |
| loss_iid | L3 (0.7) | 0.731 ± 0.009 | 0.916 | 0.788 | 0.46 | above_floor |
| loss_iid | L4 (0.9) | 0.683 ± 0.002 | 0.909 | 0.743 | 0.17 | above_floor |
| pose | L0 (0.2) | 0.697 ± 0.010 | 0.846 | 0.810 | 1.59 | above_floor |
| pose | L1 (0.4) | 0.448 ± 0.014 | 0.683 | 0.624 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.304 ± 0.008 | 0.582 | 0.474 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.334 ± 0.008 | 0.636 | 0.479 | 1.55 | below_floor |
| pose | L4 (3.2) | 0.443 ± 0.012 | 0.746 | 0.559 | 1.31 | below_floor |
| stale | L0 (2) | 0.630 ± 0.001 | 0.804 | 0.773 | 1.59 | above_floor |
| stale | L1 (4) | 0.406 ± 0.000 | 0.636 | 0.612 | 1.59 | below_floor |
| stale | L2 (8) | 0.326 ± 0.000 | 0.536 | 0.561 | 1.59 | below_floor |
| stale | L3 (16) | 0.292 ± 0.001 | 0.490 | 0.547 | 1.58 | below_floor |
| stale | L4 (32) | 0.277 ± 0.000 | 0.473 | 0.537 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.793 ± 0.013 | 0.878 | 0.879 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.663 ± 0.004 | 0.775 | 0.814 | 1.59 | above_floor |
| swap | L2 (0.5) | 0.539 ± 0.006 | 0.679 | 0.735 | 1.59 | below_floor |
| swap | L3 (0.75) | 0.400 ± 0.005 | 0.562 | 0.625 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.266 ± 0.003 | 0.442 | 0.491 | 1.59 | below_floor |

## early

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| ghosts | L0 (1) | 0.770 ± 0.002 | 0.824 | 0.893 | n/a | above_floor |
| ghosts | L1 (2) | 0.742 ± 0.002 | 0.793 | 0.888 | n/a | above_floor |
| ghosts | L2 (4) | 0.690 ± 0.001 | 0.735 | 0.878 | n/a | above_floor |
| ghosts | L3 (8) | 0.613 ± 0.004 | 0.648 | 0.864 | n/a | above_floor |
| ghosts | L4 (16) | 0.499 ± 0.002 | 0.527 | 0.826 | n/a | below_floor |
| latency | L0 (1) | 0.351 ± 0.000 | 0.566 | 0.588 | n/a | below_floor |
| latency | L1 (2) | 0.195 ± 0.000 | 0.412 | 0.415 | n/a | below_floor |
| latency | L2 (4) | 0.178 ± 0.000 | 0.372 | 0.393 | n/a | below_floor |
| latency | L3 (6) | 0.213 ± 0.000 | 0.386 | 0.466 | n/a | below_floor |
| latency | L4 (8) | 0.228 ± 0.000 | 0.390 | 0.506 | n/a | below_floor |
| latency | L5 (10) | 0.238 ± 0.000 | 0.395 | 0.532 | n/a | below_floor |
| loss_burst | L0 (0.1) | 0.790 ± 0.002 | 0.858 | 0.883 | n/a | above_floor |
| loss_burst | L1 (0.3) | 0.745 ± 0.005 | 0.847 | 0.847 | n/a | above_floor |
| loss_burst | L2 (0.5) | 0.704 ± 0.002 | 0.838 | 0.810 | n/a | above_floor |
| loss_burst | L3 (0.7) | 0.641 ± 0.003 | 0.823 | 0.752 | n/a | above_floor |
| loss_burst | L4 (0.9) | 0.623 ± 0.005 | 0.817 | 0.737 | n/a | above_floor |
| loss_iid | L0 (0.1) | 0.788 ± 0.001 | 0.855 | 0.883 | n/a | above_floor |
| loss_iid | L1 (0.3) | 0.753 ± 0.003 | 0.849 | 0.852 | n/a | above_floor |
| loss_iid | L2 (0.5) | 0.702 ± 0.007 | 0.838 | 0.807 | n/a | above_floor |
| loss_iid | L3 (0.7) | 0.640 ± 0.004 | 0.820 | 0.750 | n/a | above_floor |
| loss_iid | L4 (0.9) | 0.579 ± 0.003 | 0.802 | 0.696 | n/a | at_floor |
| pose | L0 (0.2) | 0.572 ± 0.009 | 0.727 | 0.742 | n/a | at_floor |
| pose | L1 (0.4) | 0.276 ± 0.004 | 0.528 | 0.468 | n/a | below_floor |
| pose | L2 (0.8) | 0.153 ± 0.006 | 0.430 | 0.290 | n/a | below_floor |
| pose | L3 (1.6) | 0.170 ± 0.003 | 0.497 | 0.287 | n/a | below_floor |
| pose | L4 (3.2) | 0.263 ± 0.005 | 0.600 | 0.396 | n/a | below_floor |
| stale | L0 (2) | 0.548 ± 0.000 | 0.712 | 0.742 | n/a | below_floor |
| stale | L1 (4) | 0.337 ± 0.001 | 0.554 | 0.567 | n/a | below_floor |
| stale | L2 (8) | 0.260 ± 0.000 | 0.461 | 0.503 | n/a | below_floor |
| stale | L3 (16) | 0.243 ± 0.000 | 0.422 | 0.515 | n/a | below_floor |
| stale | L4 (32) | 0.230 ± 0.000 | 0.406 | 0.512 | n/a | below_floor |
| swap | L0 (0.1) | 0.728 ± 0.011 | 0.807 | 0.855 | n/a | above_floor |
| swap | L1 (0.3) | 0.554 ± 0.003 | 0.690 | 0.749 | n/a | below_floor |
| swap | L2 (0.5) | 0.415 ± 0.009 | 0.586 | 0.641 | n/a | below_floor |
| swap | L3 (0.75) | 0.278 ± 0.008 | 0.467 | 0.510 | n/a | below_floor |
| swap | L4 (1.0) | 0.166 ± 0.004 | 0.346 | 0.366 | n/a | below_floor |

## fcooper

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| bandwidth | L0 (16) | 0.790 ± 0.000 | 0.876 | 0.874 | 1.59 | above_floor |
| bandwidth | L1 (8) | 0.788 ± 0.001 | 0.875 | 0.873 | 1.59 | above_floor |
| bandwidth | L2 (4) | 0.754 ± 0.001 | 0.858 | 0.856 | 1.59 | above_floor |
| bandwidth | L3 (2) | 0.476 ± 0.001 | 0.712 | 0.637 | 1.59 | below_floor |
| bandwidth | L4 (1) | 0.365 ± 0.000 | 0.584 | 0.543 | 1.59 | below_floor |
| ghosts | L0 (1) | 0.768 ± 0.001 | 0.853 | 0.871 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.749 ± 0.001 | 0.830 | 0.869 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.710 ± 0.003 | 0.788 | 0.863 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.652 ± 0.001 | 0.721 | 0.852 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.559 ± 0.002 | 0.617 | 0.831 | 1.59 | at_floor |
| latency | L0 (1) | 0.352 ± 0.001 | 0.586 | 0.574 | 1.59 | below_floor |
| latency | L1 (2) | 0.198 ± 0.000 | 0.441 | 0.407 | 1.59 | below_floor |
| latency | L2 (4) | 0.179 ± 0.000 | 0.405 | 0.379 | 1.59 | below_floor |
| latency | L3 (6) | 0.194 ± 0.000 | 0.414 | 0.401 | 1.59 | below_floor |
| latency | L4 (8) | 0.198 ± 0.000 | 0.419 | 0.406 | 1.59 | below_floor |
| latency | L5 (10) | 0.199 ± 0.000 | 0.420 | 0.410 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.777 ± 0.001 | 0.872 | 0.864 | 1.43 | above_floor |
| loss_burst | L1 (0.3) | 0.749 ± 0.004 | 0.865 | 0.837 | 1.14 | above_floor |
| loss_burst | L2 (0.5) | 0.699 ± 0.004 | 0.850 | 0.796 | 0.79 | above_floor |
| loss_burst | L3 (0.7) | 0.652 ± 0.006 | 0.833 | 0.755 | 0.47 | above_floor |
| loss_burst | L4 (0.9) | 0.634 ± 0.004 | 0.827 | 0.740 | 0.36 | above_floor |
| loss_iid | L0 (0.1) | 0.776 ± 0.001 | 0.872 | 0.861 | 1.42 | above_floor |
| loss_iid | L1 (0.3) | 0.747 ± 0.002 | 0.865 | 0.836 | 1.12 | above_floor |
| loss_iid | L2 (0.5) | 0.700 ± 0.002 | 0.851 | 0.796 | 0.79 | above_floor |
| loss_iid | L3 (0.7) | 0.651 ± 0.005 | 0.834 | 0.754 | 0.47 | above_floor |
| loss_iid | L4 (0.9) | 0.595 ± 0.003 | 0.812 | 0.707 | 0.16 | at_floor |
| pose | L0 (0.2) | 0.591 ± 0.016 | 0.761 | 0.750 | 1.59 | at_floor |
| pose | L1 (0.4) | 0.319 ± 0.007 | 0.550 | 0.529 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.172 ± 0.001 | 0.406 | 0.347 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.170 ± 0.005 | 0.448 | 0.313 | 1.55 | below_floor |
| pose | L4 (3.2) | 0.269 ± 0.009 | 0.591 | 0.405 | 1.32 | below_floor |
| stale | L0 (2) | 0.545 ± 0.000 | 0.732 | 0.723 | 1.59 | below_floor |
| stale | L1 (4) | 0.335 ± 0.000 | 0.578 | 0.552 | 1.59 | below_floor |
| stale | L2 (8) | 0.254 ± 0.000 | 0.492 | 0.470 | 1.59 | below_floor |
| stale | L3 (16) | 0.227 ± 0.000 | 0.458 | 0.443 | 1.58 | below_floor |
| stale | L4 (32) | 0.203 ± 0.000 | 0.433 | 0.422 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.697 ± 0.008 | 0.823 | 0.812 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.521 ± 0.010 | 0.721 | 0.678 | 1.59 | below_floor |
| swap | L2 (0.5) | 0.374 ± 0.007 | 0.619 | 0.556 | 1.59 | below_floor |
| swap | L3 (0.75) | 0.249 ± 0.007 | 0.502 | 0.425 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.139 ± 0.002 | 0.364 | 0.293 | 1.59 | below_floor |

## late

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| ghosts | L0 (1) | 0.767 ± 0.001 | 0.827 | 0.870 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.755 ± 0.000 | 0.811 | 0.869 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.731 ± 0.001 | 0.779 | 0.868 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.687 ± 0.002 | 0.726 | 0.865 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.626 ± 0.002 | 0.648 | 0.860 | 1.59 | above_floor |
| latency | L0 (1) | 0.326 ± 0.000 | 0.541 | 0.557 | 1.59 | below_floor |
| latency | L1 (2) | 0.221 ± 0.000 | 0.429 | 0.451 | 1.59 | below_floor |
| latency | L2 (4) | 0.267 ± 0.000 | 0.425 | 0.551 | 1.59 | below_floor |
| latency | L3 (6) | 0.307 ± 0.000 | 0.438 | 0.625 | 1.59 | below_floor |
| latency | L4 (8) | 0.311 ± 0.000 | 0.437 | 0.638 | 1.59 | below_floor |
| latency | L5 (10) | 0.310 ± 0.000 | 0.434 | 0.647 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.770 ± 0.002 | 0.847 | 0.857 | 1.42 | above_floor |
| loss_burst | L1 (0.3) | 0.741 ± 0.004 | 0.846 | 0.828 | 1.14 | above_floor |
| loss_burst | L2 (0.5) | 0.710 ± 0.010 | 0.845 | 0.796 | 0.81 | above_floor |
| loss_burst | L3 (0.7) | 0.667 ± 0.003 | 0.842 | 0.752 | 0.52 | above_floor |
| loss_burst | L4 (0.9) | 0.640 ± 0.003 | 0.835 | 0.727 | 0.37 | above_floor |
| loss_iid | L0 (0.1) | 0.768 ± 0.004 | 0.846 | 0.856 | 1.43 | above_floor |
| loss_iid | L1 (0.3) | 0.740 ± 0.004 | 0.846 | 0.826 | 1.11 | above_floor |
| loss_iid | L2 (0.5) | 0.706 ± 0.006 | 0.845 | 0.794 | 0.80 | above_floor |
| loss_iid | L3 (0.7) | 0.660 ± 0.003 | 0.839 | 0.746 | 0.48 | above_floor |
| loss_iid | L4 (0.9) | 0.610 ± 0.002 | 0.832 | 0.696 | 0.17 | above_floor |
| pose | L0 (0.2) | 0.507 ± 0.002 | 0.670 | 0.691 | 1.59 | below_floor |
| pose | L1 (0.4) | 0.269 ± 0.006 | 0.463 | 0.505 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.215 ± 0.004 | 0.377 | 0.473 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.297 ± 0.013 | 0.449 | 0.580 | 1.59 | below_floor |
| pose | L4 (3.2) | 0.401 ± 0.014 | 0.571 | 0.643 | 1.59 | below_floor |
| stale | L0 (2) | 0.522 ± 0.000 | 0.691 | 0.712 | 1.59 | below_floor |
| stale | L1 (4) | 0.349 ± 0.000 | 0.548 | 0.586 | 1.59 | below_floor |
| stale | L2 (8) | 0.320 ± 0.000 | 0.484 | 0.594 | 1.59 | below_floor |
| stale | L3 (16) | 0.312 ± 0.000 | 0.455 | 0.618 | 1.58 | below_floor |
| stale | L4 (32) | 0.298 ± 0.000 | 0.440 | 0.618 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.695 ± 0.019 | 0.760 | 0.852 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.576 ± 0.012 | 0.628 | 0.815 | 1.59 | at_floor |
| swap | L2 (0.5) | 0.471 ± 0.001 | 0.522 | 0.774 | 1.59 | below_floor |
| swap | L3 (0.75) | 0.375 ± 0.007 | 0.428 | 0.715 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.275 ± 0.003 | 0.339 | 0.639 | 1.59 | below_floor |

## v2vnet

| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |
|---|---|---|---|---|---|---|
| bandwidth | L0 (16) | 0.822 ± 0.000 | 0.886 | 0.913 | 1.59 | above_floor |
| bandwidth | L1 (8) | 0.822 ± 0.000 | 0.887 | 0.912 | 1.59 | above_floor |
| bandwidth | L2 (4) | 0.807 ± 0.000 | 0.865 | 0.910 | 1.59 | above_floor |
| bandwidth | L3 (2) | 0.700 ± 0.001 | 0.797 | 0.852 | 1.59 | above_floor |
| bandwidth | L4 (1) | 0.594 ± 0.001 | 0.758 | 0.768 | 1.59 | at_floor |
| ghosts | L0 (1) | 0.816 ± 0.001 | 0.881 | 0.910 | 1.59 | above_floor |
| ghosts | L1 (2) | 0.811 ± 0.002 | 0.877 | 0.909 | 1.59 | above_floor |
| ghosts | L2 (4) | 0.800 ± 0.002 | 0.868 | 0.906 | 1.59 | above_floor |
| ghosts | L3 (8) | 0.776 ± 0.001 | 0.849 | 0.897 | 1.59 | above_floor |
| ghosts | L4 (16) | 0.746 ± 0.003 | 0.825 | 0.884 | 1.59 | above_floor |
| latency | L0 (1) | 0.487 ± 0.000 | 0.680 | 0.698 | 1.59 | below_floor |
| latency | L1 (2) | 0.246 ± 0.000 | 0.469 | 0.479 | 1.59 | below_floor |
| latency | L2 (4) | 0.206 ± 0.000 | 0.414 | 0.431 | 1.59 | below_floor |
| latency | L3 (6) | 0.243 ± 0.000 | 0.433 | 0.496 | 1.59 | below_floor |
| latency | L4 (8) | 0.257 ± 0.000 | 0.438 | 0.529 | 1.59 | below_floor |
| latency | L5 (10) | 0.261 ± 0.000 | 0.437 | 0.545 | 1.58 | below_floor |
| loss_burst | L0 (0.1) | 0.807 ± 0.001 | 0.879 | 0.902 | 1.44 | above_floor |
| loss_burst | L1 (0.3) | 0.773 ± 0.006 | 0.863 | 0.877 | 1.12 | above_floor |
| loss_burst | L2 (0.5) | 0.724 ± 0.008 | 0.841 | 0.842 | 0.78 | above_floor |
| loss_burst | L3 (0.7) | 0.677 ± 0.006 | 0.818 | 0.807 | 0.49 | above_floor |
| loss_burst | L4 (0.9) | 0.648 ± 0.002 | 0.804 | 0.786 | 0.35 | above_floor |
| loss_iid | L0 (0.1) | 0.805 ± 0.001 | 0.878 | 0.901 | 1.43 | above_floor |
| loss_iid | L1 (0.3) | 0.768 ± 0.001 | 0.860 | 0.875 | 1.12 | above_floor |
| loss_iid | L2 (0.5) | 0.728 ± 0.002 | 0.842 | 0.845 | 0.81 | above_floor |
| loss_iid | L3 (0.7) | 0.674 ± 0.003 | 0.818 | 0.806 | 0.49 | above_floor |
| loss_iid | L4 (0.9) | 0.609 ± 0.003 | 0.784 | 0.755 | 0.15 | above_floor |
| pose | L0 (0.2) | 0.702 ± 0.008 | 0.818 | 0.838 | 1.59 | above_floor |
| pose | L1 (0.4) | 0.490 ± 0.005 | 0.689 | 0.687 | 1.59 | below_floor |
| pose | L2 (0.8) | 0.283 ± 0.002 | 0.539 | 0.484 | 1.59 | below_floor |
| pose | L3 (1.6) | 0.278 ± 0.003 | 0.568 | 0.446 | 1.54 | below_floor |
| pose | L4 (3.2) | 0.363 ± 0.013 | 0.654 | 0.525 | 1.33 | below_floor |
| stale | L0 (2) | 0.641 ± 0.000 | 0.785 | 0.806 | 1.59 | above_floor |
| stale | L1 (4) | 0.401 ± 0.000 | 0.614 | 0.629 | 1.59 | below_floor |
| stale | L2 (8) | 0.304 ± 0.000 | 0.511 | 0.548 | 1.59 | below_floor |
| stale | L3 (16) | 0.279 ± 0.000 | 0.469 | 0.547 | 1.58 | below_floor |
| stale | L4 (32) | 0.260 ± 0.000 | 0.449 | 0.536 | 1.57 | below_floor |
| swap | L0 (0.1) | 0.753 ± 0.010 | 0.839 | 0.877 | 1.59 | above_floor |
| swap | L1 (0.3) | 0.624 ± 0.005 | 0.745 | 0.798 | 1.59 | above_floor |
| swap | L2 (0.5) | 0.499 ± 0.017 | 0.655 | 0.708 | 1.59 | below_floor |
| swap | L3 (0.75) | 0.360 ± 0.006 | 0.550 | 0.590 | 1.59 | below_floor |
| swap | L4 (1.0) | 0.235 ± 0.003 | 0.438 | 0.449 | 1.59 | below_floor |
