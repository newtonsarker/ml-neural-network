#!/usr/bin/env python

"""Tests for `ml_neural_network` package."""

import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# from ml_neural_network.ml_neural_network import add_numbers
import ml_neural_network.ml_neural_network as nn

def test_add_numbers():
    """Test the add_numbers function."""
    assert nn.add_numbers(1, 2) == 3
    assert nn.add_numbers(-1, 1) == 0
    assert nn.add_numbers(0, 0) == 0
    assert nn.add_numbers(1.5, 2.5) == 4.0
