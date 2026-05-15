# genz-classifier

A Python package for computing customer embeddings from purchase history. It aggregates per-product (ASIN) embedding vectors weighted by each customer's purchase frequency to produce a single dense vector representation per customer, suitable for downstream classification, segmentation, or recommendation tasks.

## Features

- **`features.py`** — Core computations:
  - `compute_purchase_frequencies(purchase_records)` — turns a list of raw purchase records into an ASIN → frequency map.
  - `compute_customer_embedding(asin_embeddings, customer_purchase_frequency)` — produces a frequency-weighted average embedding for one customer.
- **`pipeline.py`** — End-to-end helper:
  - `compute_customer_embedding_from_records(purchase_records, asin_embeddings)` — goes directly from raw records to a customer embedding in one call.
- **`main.py`** — A FastAPI service exposing an `/embed` endpoint for computing embeddings over HTTP.

## Installation

Install from source:

```bash
git clone https://github.com/your-org/genz-classifier.git
cd genz-classifier
pip install -e .
```

Or install directly:

```bash
pip install genz-classifier
```

Requires Python 3.10+ (uses `list[Dict[...]]` syntax).

## Usage

### Library usage

```python
import numpy as np
from genz_classifier.pipeline import compute_customer_embedding_from_records

purchase_records = [
    {"asin": "B001", "date": "2024-01-15"},
    {"asin": "B001", "date": "2024-02-03"},
    {"asin": "B002", "date": "2024-02-20"},
]

asin_embeddings = {
    "B001": np.array([0.1, 0.2, 0.3]),
    "B002": np.array([0.4, 0.5, 0.6]),
}

embedding = compute_customer_embedding_from_records(purchase_records, asin_embeddings)
print(embedding)  # frequency-weighted average vector
```

You can also use the lower-level building blocks directly:

```python
from genz_classifier.features import (
    compute_purchase_frequencies,
    compute_customer_embedding,
)

frequencies = compute_purchase_frequencies(purchase_records)
# {"B001": 2, "B002": 1}

embedding = compute_customer_embedding(asin_embeddings, frequencies)
```

### REST API

Start the FastAPI server:

```bash
uvicorn genz_classifier.main:app --host 0.0.0.0 --port 8000
```

Call the `/embed` endpoint:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "purchase_records": [
      {"asin": "B001", "date": "2024-01-15"},
      {"asin": "B002", "date": "2024-02-03"}
    ],
    "asin_embeddings": {
      "B001": [0.1, 0.2, 0.3],
      "B002": [0.4, 0.5, 0.6]
    }
  }'
```

Interactive API docs are available at `http://localhost:8000/docs`.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

For coverage:

```bash
pytest --cov=genz_classifier --cov-report=term-missing
```

## Running with Docker

Build the image:

```bash
docker build -t genz-classifier .
```

Run the API:

```bash
docker run -p 8000:8000 genz-classifier
```

The service will be available at `http://localhost:8000`, with interactive docs at `/docs`.

## License

See `LICENSE` for details.
