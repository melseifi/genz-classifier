import numpy as np
import pytest
from genz_classifier.features import compute_customer_embedding
from genz_classifier.features import compute_purchase_frequencies

def test_weighted_average_basic():
    asin_embeddings = {"A001": np.array([1.0, 0.0, 0.0]), "A002": np.array([0.0, 1.0, 0.0])}
    frequency = {"A001":2, "A002":2}
    customer_embedding = compute_customer_embedding(asin_embeddings, frequency)
    np.testing.assert_array_almost_equal(customer_embedding, [0.5, 0.5, 0])



def test_empty_frequency_raises_error():
    asin_embeddings = {"A001": np.array([1.0, 0.0, 0.0]), "A002": np.array([0.0, 1.0, 0.0])}
    frequency = {"A001": 0, "A002": 0}

    with pytest.raises(ValueError):
        compute_customer_embedding(asin_embeddings, frequency)

def test_missing_asin_embedding():
    asin_embeddings = {"A001": np.array([1.0, 0.0, 0.0])}
    frequency = {"A001":2, "A002":2}
    customer_embedding = compute_customer_embedding(asin_embeddings, frequency)
    np.testing.assert_array_almost_equal(customer_embedding, [1.0, 0.0, 0.0])

def test_asin_frequency():
    purchases = [{"asin": "B001", "date": "2024-01-15"},
                {"asin": "B001", "date": "2024-02-03"},
                 {"asin": "B002", "date": "2024-02-03"}]
    freq = compute_purchase_frequencies(purchases)
    assert (freq== {"B001":2, "B002":1})

def test_empty_purchases_raises_error():
    purchases = []
    with pytest.raises(ValueError):
        compute_purchase_frequencies(purchases)

def test_asin_single_purchase():
    purchases = [{"asin": "B001", "date": "2024-01-15"}]
    freq = compute_purchase_frequencies(purchases)
    assert(freq == {"B001":1})