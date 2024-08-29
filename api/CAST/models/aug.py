import torch as th
import numpy as np
import dgl

# __func is original, func is GPU optimized
def __random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

def __drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def __mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    """
    Given a graph, randomly drops features and masks edges.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    x : torch.Tensor
        The input features.
    feat_drop_rate : float
        The probability of dropping a feature.
    edge_mask_rate : float
        The probability of masking an edge.
    
    Returns
    -------
    DGLGraph
        The graph after randomly masking edges.
    torch.Tensor
        The features after randomly dropping features.
    """

    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = x.clone()
    feat = drop_feature(feat, feat_drop_rate)

    ng = dgl.graph([], device=graph.device)
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

def drop_feature(x, drop_prob):
    """
    Randomly drop features with probability drop_prob

    Parameters
    ----------
    x : torch.Tensor
        The input features.
    drop_prob : float
        The probability of dropping a feature.
    
    Returns
    -------
    torch.Tensor
        The remaining features after random dropping.
    """
    # boolean mask for dropping features with probability drop_prob
    drop_mask = th.empty( 
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    # x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    """
    Randomly mask edges with probability mask_prob.

    Parameters
    ----------
    graph : DGLGraph
        The input graph (only used to take the number of edges).
    mask_prob : float
        The probability of masking an edge.
    
    Returns
    -------
    torch.Tensor
        A 1D tensor of indices of the remaining edges after random masking.
    """

    E = graph.number_of_edges()
    mask_rates = th.ones(E, device=graph.device) * mask_prob
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx