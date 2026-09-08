Overview
========

Import style
------------

`common_utils` exposes many utilities. You can either import from the top-level package:

.. code-block:: python

   from common_utils import load_mask, save_mask, mask_iou

or import from specific submodules for clarity:

.. code-block:: python

   from common_utils.mask_utils import load_mask, save_mask, mask_iou


Where to look
-------------

Utilities are grouped roughly by domain:

- Data IO & processing: ``common_utils.data`` (arrays, normalization, image/video helpers, LMDB, save/load).
- Distributed helpers: ``common_utils.distributed``.
- Filesystem helpers: ``common_utils.fs``.
- Inference helpers: ``common_utils.inference``.
- Misc utilities: top-level modules like ``common_utils.timer``, ``common_utils.mask_utils``.


Optional extras
--------------------

Some inference utilities require optional dependencies:

- Ray helpers: from a local checkout, install with ``pip install ".[ray]"``.
- Ollama helpers: from a local checkout, install with ``pip install ".[ollama]"``.

If you only need the core utilities, use ``pip install .`` from a local checkout
or follow :doc:`installation` to install from GitHub.
