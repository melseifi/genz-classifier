import numpy as np
from typing import Dict

def compute_customer_embedding(
    asin_embeddings: Dict[str, np.ndarray],
    customer_purchase_frequency: Dict[str, int]
) -> np.ndarray:
    """
    Function calculates the average embedding vector for a single customerId using the
    ASIN embedding vectors for each purchase and the frequency of each purchase.

    Args:
        asin_embeddings: dictionary with key = ASIN, value = embedding vector
        customer_purchase_frequency: dictionary with key = ASIN, value = purchase frequency

    Returns:
        customer_embeddings:  customer embedding vector
    """
    if not customer_purchase_frequency:
        raise ValueError("customer_purchase_frequency cannot be empty")

    if sum(customer_purchase_frequency.values()) == 0:
        raise ValueError("total purchase frequency cannot be zero")


    embed = list(asin_embeddings.values())
    embed_len = len(embed[0])

    total_embeddings = np.zeros(embed_len)
    total_purchases = 0

    for asin, frequency in customer_purchase_frequency.items():
        if asin in asin_embeddings:
            total_purchases += frequency
            total_embeddings+=asin_embeddings[asin]*frequency

    if total_purchases == 0:
        raise ValueError("no valid ASINs found in asin_embeddings")

    avg_embeddings= total_embeddings / total_purchases

    return avg_embeddings
