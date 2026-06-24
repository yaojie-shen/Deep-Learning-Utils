# -*- coding: utf-8 -*-
# @Time    : 3/20/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_mask_utils.py

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image

from dl_utils import (  # noqa: E402
    intersect_masks,
    invert_mask,
    load_mask,
    mask_iou,
    save_mask,
    subtract_mask,
    union_masks,
)


def test_read_save_mask_roundtrip_with_tmp_path(tmp_path):
    p = tmp_path / "m.png"

    m = np.zeros((16, 16), dtype=bool)
    m[1:4, 2:6] = True

    save_mask(m, p)
    m2 = load_mask(p)
    assert m2.dtype == np.bool_
    assert m2.shape == m.shape
    assert np.array_equal(m2, m)

    # --- uint8 output ---
    m3 = load_mask(p, as_bool=False)
    assert m3.dtype == np.uint8
    assert set(np.unique(m3)).issubset({0, 255})


def test_mask_set_ops():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)

    a[1:4, 1:4] = True
    b[3:6, 3:6] = True

    u = union_masks(a, b)
    i = intersect_masks(a, b)
    d = subtract_mask(a, b)
    inv = invert_mask(a)

    assert u.sum() == (a | b).sum()
    assert i.sum() == (a & b).sum()
    assert np.array_equal(d, a & (~b))
    assert np.array_equal(inv, ~a)


def test_mask_set_ops_accept_uint8_and_pil_inputs():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[1:4, 1:4] = True
    b[3:6, 3:6] = True

    a_u8 = a.astype(np.uint8) * 255
    b_u8 = b.astype(np.uint8) * 255
    b_img = Image.fromarray(b_u8)

    u = union_masks(a_u8, b_img)
    assert u.dtype == np.bool_
    assert np.array_equal(u, a | b)


def test_mask_iou_basic_and_empty_cases():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[1:4, 1:4] = True
    b[3:6, 3:6] = True

    expected = float((a & b).sum()) / float((a | b).sum())
    assert mask_iou(a, b) == expected

    empty = np.zeros((8, 8), dtype=bool)
    assert mask_iou(empty, empty) == 1.0


def test_mask_iou_accept_uint8_and_pil_inputs():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[1:4, 1:4] = True
    b[3:6, 3:6] = True

    a_u8 = a.astype(np.uint8) * 255
    b_img = Image.fromarray(b.astype(np.uint8) * 255)

    assert mask_iou(a_u8, b_img) == mask_iou(a, b)
