# -*- coding: utf-8 -*-
# @Time    : 8/1/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_inspect_data_utils.py

import numpy as np
import pandas as pd
import pytest
import torch

from common_utils import inspect_data


@pytest.mark.parametrize(
    "test_data",
    [
        {"a": 1, "b": 2, "c": 3},
        [{"x": i, "y": i * 2} for i in range(20)],
        torch.randn(3, 4),
        np.random.rand(2, 3),
        pd.DataFrame({"id": [1, 2, 3], "label": ["cat", "dog", "fish"]}),
        {
            "nested": {
                "list": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
                "config": {"lr": 0.001, "batch_size": 32},
                "notes": "A" * 150,
            },
            "meta": {"created": "2025-08-01", "version": 1.0},
            "tuple": (1, 2, 3),
            "tuple_of_mixed_types": (1, 2.0, "A"),
            "tuple_of_log_string": ("A" * 1000, "B" * 100, 1)
        }
    ]
)
def test_inspect_data_visualize(test_data):
    """Test inspect_data with various data types for visual output only."""
    print("\n\n🔍 Visualizing structure for test data:")
    inspect_data(test_data, name="data_inspect")
