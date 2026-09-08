# Deep-Learning-Utils

`dl-utils` is a lightweight Python utility library for deep learning workflows. It includes helpers for data IO and normalization, filesystem traversal, distributed training, inference utilities, data inspection, and small general-purpose utilities.

For the full API reference and detailed documentation, please refer to the [documentation](https://yaojie-shen.github.io/Deep-Learning-Utils/).

## Installation

Install the package from source:

```bash
pip install git+https://github.com/yaojie-shen/Deep-Learning-Utils.git
```

Install the latest development version:

```bash
pip install git+https://github.com/yaojie-shen/Deep-Learning-Utils.git@dev
```

For local development:

```bash
pip install -e .
```

Optional extras are available for some features:

```bash
pip install "dl-utils[ray]"      # Ray inference helpers
```

## Usage

It is recommended to import most utilities directly from the top-level `dl_utils` package:

```python
from dl_utils import save_json, load_json, inspect_data

data = {"name": "example", "items": [1, 2, 3]}

save_json(data, "output/example.json", save_pretty=True)
loaded = load_json("output/example.json")

inspect_data(loaded)
```

## API Documentation

Detailed API documentation is maintained in the project docs rather than duplicated here:

- Documentation home: <https://yaojie-shen.github.io/Deep-Learning-Utils/>
- API reference: <https://yaojie-shen.github.io/Deep-Learning-Utils/api_reference.html>

## Development

Run tests with:

```bash
pytest
```

Build documentation locally with:

```bash
pip install -e ".[docs]"
make -C docs html
```
