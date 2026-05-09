import numpy as np
import pytest
from genz_classifier.pipeline import compute_customer_embedding_from_records

def test_pipeline_basic():
    asin_embeddings = {"A001": np.array([1.0, 0.0, 0.0]), "A002": np.array([0.0, 1.0, 0.0])}
    purchases = [{"asin": "A001", "date": "2024-01-15"},
                 {"asin": "A001", "date": "2024-02-03"},
                 {"asin": "A002", "date": "2024-02-03"},
                 {"asin": "A002", "date": "2024-02-07"}]
    customer_embedding = compute_customer_embedding_from_records(purchases,asin_embeddings)
    np.testing.assert_array_almost_equal(customer_embedding, [0.5, 0.5, 0])

def test_empty_purchases_raises_error():
    purchases = []
    asin_embeddings = {"A001": np.array([1.0, 0.0, 0.0]), "A002": np.array([0.0, 1.0, 0.0])}
    with pytest.raises(ValueError):
        compute_customer_embedding_from_records(purchases, asin_embeddings)