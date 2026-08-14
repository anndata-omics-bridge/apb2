"""Exact in-memory matrix types accepted by backend-neutral calculations."""

from __future__ import annotations

from typing import TypeIs

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse

type DenseQuantMatrix = NDArray[np.float32] | NDArray[np.float64]
type CompressedSparseMatrix[ScalarT: (np.float32, np.float64)] = (
    csr_matrix[ScalarT] | csc_matrix[ScalarT] | csr_array[ScalarT] | csc_array[ScalarT]
)
type SparseQuantMatrix = CompressedSparseMatrix[np.float32] | CompressedSparseMatrix[np.float64]
type QuantMatrix = DenseQuantMatrix | SparseQuantMatrix


def is_sparse_matrix(value: object) -> TypeIs[SparseQuantMatrix]:
    """Narrow a quantitative matrix to the supported SciPy containers."""
    return issparse(value) and getattr(value, "format", "") in {"csr", "csc"}
