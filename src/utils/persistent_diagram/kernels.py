from typing import List

import numpy as np
import gudhi


def bottleneck_distance(
        dg1: np.ndarray,
        dg2: np.ndarray,
        epsilon=1e-3):
    return gudhi.bottleneck_distance(dg1, dg2, epsilon)


def bottleneck_kernel(
        dg1: List[np.ndarray],
        dg2: List[np.ndarray] = None,
        epsilon=1e-3):
    num_diag1 = len(dg1)

    if dg2 is None:
        matrix = np.zeros((num_diag1, num_diag1))
        for i in range(num_diag1):
            # sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
            for j in range(i + 1, num_diag1):
                matrix[i, j] = bottleneck_distance(dg1[i], dg1[j], epsilon)
                matrix[j, i] = matrix[i, j]
    else:
        num_diag2 = len(dg2)
        matrix = np.zeros((num_diag1, num_diag2))
        for i in range(num_diag1):
            # sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
            for j in range(num_diag2):
                matrix[i, j] = bottleneck_distance(dg1[i], dg2[j], epsilon)

    return matrix


def sliced_wasserstein_kernel():
    pass


