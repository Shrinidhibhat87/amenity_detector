"""Python file that contains the retriver function for RAG based pipeline."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

def find_closest_entry(
    query_embedding: np.ndarray,
    metadata: List[Dict[str, Any]],
    top_k: int = 1,
):
    """
    Find the closest entry in the metadata based on cosine similarity.

    Args:
        query_embedding (np.ndarray): The embedding of the query.
        metadata (List[Dict[str, Any]]): List of metadata entries.
        top_k (int): Number of closest entries to return.

    Returns:
        List[Dict[str, Any]]: List of closest metadata entries.
    """
    # Open an empty similar list
    similar_list = []

    for entry in metadata:
        if 'embedding' not in entry:
            continue
        # Calculate the cosine similarity
        similarity = cosine_similarity(query_embedding.reshape(1, -1), entry['embedding'].reshape(1, -1))[0][0]
        # Append the similarity and entry to the list
        similar_list.append((similarity, entry))
    
    # Once we have a list, sort it by similarity
    similar_list.sort(reverse=True, key=lambda x: x[0])

    # Return the top k entries
    return [entry for _, entry in similar_list[:top_k]]
