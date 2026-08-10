"""
Tests for fluxq.solvers.brute_force.

This module was extracted from logic that was previously duplicated
identically in demos/toy_brute_force.py and demos/rts_gmlc_5gen_brute_force.py.
These tests exist to confirm the extraction preserved correctness exactly,
and to give the module its own regression coverage as FluxQ's ground-truth
solver.

Run with:
    python -m pytest tests/test_brute_force.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.solvers.brute_force import brute_force


class TestBruteForce:
    def test_finds_true_minimum_simple_case(self):
        """
        Q = [[1, -3], [0, 1]] over x in {0,1}^2:
          00 -> 0, 10 -> 1, 01 -> 1, 11 -> 1-3+1 = -1
        True minimum is -1 at x=[1,1].
        """
        Q = np.array([[1.0, -3.0], [0.0, 1.0]])
        results = brute_force(Q, n_vars=2, top_k=4)
        best_energy, best_x = results[0]
        assert abs(best_energy - (-1.0)) < 1e-9
        assert np.array_equal(best_x, np.array([1.0, 1.0]))

    def test_results_sorted_ascending(self):
        Q = np.array([[1.0, -3.0], [0.0, 1.0]])
        results = brute_force(Q, n_vars=2, top_k=4)
        energies = [e for e, _ in results]
        assert energies == sorted(energies)

    def test_top_k_limits_results(self):
        Q = np.zeros((4, 4))
        results = brute_force(Q, n_vars=4, top_k=3)
        assert len(results) == 3

    def test_raises_above_practical_limit(self):
        Q = np.zeros((26, 26))
        with pytest.raises(ValueError):
            brute_force(Q, n_vars=26)

    def test_all_zero_Q_gives_zero_energy_everywhere(self):
        Q = np.zeros((3, 3))
        results = brute_force(Q, n_vars=3, top_k=8)
        assert all(abs(e) < 1e-9 for e, _ in results)
        assert len(results) == 8  # all 2^3 states tied at zero
