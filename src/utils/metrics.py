import numpy as np
from sklearn.neighbors import NearestNeighbors

def category_continuity(X, labels, k=10, aggregation = 'total_mean', sigma = None):
    """
    Given an array of points X and their classes in an array of labels,
    computes the fraction of neighbors of the same class in the k nearest neighbors.
    If sigma is a float, this fraction is weighted by the neighbors distance given a Gaussian distribution.
    These scores are then aggregated with a mean od with a mean by class and then over classes.


    Arguments:
    ----------
        X: np.array
            Array of points of shape (n_samples, sample_dimension)
        labels: np.array
            Array of label for each points, of shape (n_samples,) 
        aggregation: 'total_mean' | 'by_class_mean, default 'total_mean'
            Aggregation mode
        sigma: None of float or 'auto', default None
            If not None, neighbors weights are weighted with the Gaussian distribution of distances
              of standard deviation sigma. If 'auto', the std is the median of the neighbors distances
    Returns:
    ----------
    """
    assert k > 0, f"[metrics] category continuity : the number of neighbors k must be strictly positive (received {k})."
    assert k < X.shape[0], f"[metrics] category continuity : the number of neighbors k must be strictly lower than the number of samples - 1 (received {k} > {X.shape[0] - 1})."
    # Computing neighbors (n_neighbors at k+1 because the sample itself is counted as a neighbor)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    neighbor_labels = labels[indices[:, 1:]]  # Skip the first neighbor (itself), shape (N, k)
    same_class = np.where(neighbor_labels == labels[:, np.newaxis], 1., 0)

    # Weighting neighbors weights by the distance
    if sigma is None:
        nb_class = np.sum(same_class, axis=1) / k
    else:
        sigma = np.median(dist_to_neighbors) if sigma == 'auto' else float(sigma)
        dist_to_neighbors = distances[:, 1:]
        weights = np.exp(- dist_to_neighbors ** 2 / (2 * sigma ** 2))
        # Weighting the neighbors and normalizing the weights for each point
        nb_class = np.sum(same_class * weights, axis = 1) / np.sum(weights, axis = 1)

    if aggregation == 'total_mean':
        s = np.mean(nb_class)
    elif aggregation == 'by_class_mean':
       # Mean by class, then mean across classes
        unique_classes = np.unique(labels)
        class_means = []
        for class_label in unique_classes:
            class_mask = labels == class_label
            class_mean = np.mean(nb_class[class_mask])
            class_means.append(class_mean)
        # Mean across classes
        s = np.mean(class_means)

    return s, nb_class
