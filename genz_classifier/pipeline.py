import numpy as np
from typing import Dict
from genz_classifier.features import compute_customer_embedding
from genz_classifier.features import compute_purchase_frequencies

def compute_customer_embedding_from_records(
    purchase_records: list[Dict[str, str]],
    asin_embeddings: Dict[str, np.ndarray]
) -> np.ndarray:

    """
    Function calculates the average embedding vector for a single customerId using the
    purchase records and the IEMF asin embeddings.

    Args:
        purchase_records: list of purchase records [
                                {"asin": "B001", "date": "2024-01-15"},
                                {"asin": "B001", "date": "2024-02-03"},
                            ]

        asin_embeddings: dictionary with key = ASIN, value = embedding vector

    Returns:
        customer_embeddings:  customer embedding vector
    """

    purchase_frequencies = compute_purchase_frequencies(purchase_records)
    customer_embedding = compute_customer_embedding(asin_embeddings, purchase_frequencies)
    return customer_embedding




