import torch,random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances,pairwise_distances_chunked,confusion_matrix
import scanpy as sc
from scipy.sparse import csr_matrix as csr
from utils import coords2adjacentmat

def space_project(sdata_inte, idx_source, idx_target, raw_layer,
    source_sample, target_sample, coords_source, coords_target, output_path,
    source_sample_ctype_col, target_cell_pc_feature = None, source_cell_pc_feature = None,
    k2 = 1,
    ifplot = True,
    umap_feature = 'X_umap',
    ave_dist_fold = 2,
    batch_t = '',
    alignment_shift_adjustment = 50,
    color_dict = None,
    adjust_shift = False,
    metric_t = 'cosine',
    working_memory_t = 1000
    ):
    """
    Projects the source cells to the target cells based on the k-nearest neighbors in the PCA space and phsyical distance.

    Parameters
    ----------
    sdata_inte : anndata
        The integrated anndata object.
    idx_source : np.ndarray
        The indices of the source cells.
    idx_target : np.ndarray
        The indices of the target cells.
    raw_layer : str
        The layer name of the raw data.
    source_sample : str
        The name of the source sample.
    target_sample : str
        The name of the target sample.
    coords_source : array-like
        The coordinates of the source cells.
    coords_target : array-like
        The coordinates of the target cells.
    output_path : str
        The path to save the output files.
    source_sample_ctype_col : str | None
        The column name of the cell type annotation in the source sample. If None, the projection will be performed as a single sample without cell type annotation.
    target_cell_pc_feature : array-like, optional
        The principal components of the target cells.
    source_cell_pc_feature : array-like, optional
        The principal components of the source cells. 
    k2 : int, optional (default: 1)
        The number of nearest neighbors to consider.
    ifplot : bool, optional (default: True)
        Whether to generate evaluation plots.
    umap_feature : str, optional (default: 'X_umap')
        The column name in `sdata_inte.obsm` to use for the UMAP for visualization and saving.
    ave_dist_fold : int, optional (default: 2)
        A multiplicative factor on the average distance to use for the physical distance threshold.
    batch_t : str, optional (default: '')
        The batch name used in naming the output files.
    alignment_shift_adjustment : int, optional (default: 50)
        An additive factor on the average distance for the physical distance threshold. 
    color_dict : dict, optional
        The color dictionary for visualizing the cell type annotations.
    adjust_shift : bool, optional (default: False)
        Whether to shift the coordinates of the source cells by the median shift between the target and source cells for each cell type (ignored if `source_sample_ctype_col` is not given). 
    metric_t : str | callable, optional (default: 'cosine')
        The metric to use for the pairwise distance calculations. See sklearn.metrics.pairwise_distances_chunked for more information. 
    working_memory_t : int, optional (default: 1000)
        The sought maximum memory for the chunked pairwise distance calculations.
    
    Returns
    -------
    anndata
        The integrated anndata object with the raw and normalized projected data as layers.
    list[np.ndarray, np.ndarray, np.ndarray, np.ndarray] 
       The indicies of the k-nearest neighbors, their corresponding weights, cosine distances and physical distances for each cell.
    
    """
    sdata_ref = sdata_inte[idx_target,:].copy()
    source_feat = sdata_inte[idx_source,:].layers[raw_layer].toarray()

    # Initialize output arrays
    project_ind = np.zeros([np.sum(idx_target),k2]).astype(int)
    project_weight = np.zeros_like(project_ind).astype(float)
    cdists = np.zeros_like(project_ind).astype(float)
    physical_dist = np.zeros_like(project_ind).astype(float)
    all_avg_feat = np.zeros([np.sum(idx_target),source_feat.shape[1]]).astype(float)

    if source_sample_ctype_col is not None:
        # run the projection for each cell type
        for ctype_t in np.unique(sdata_inte[idx_target].obs[source_sample_ctype_col]):
            print(f'Start to project {ctype_t} cells:')
            idx_ctype_t = np.isin(sdata_inte[idx_target].obs[source_sample_ctype_col],ctype_t)
            ave_dist_t,_,_,_ = average_dist(coords_target[idx_ctype_t,:].copy(),working_memory_t=working_memory_t)
            dist_thres = ave_dist_fold * ave_dist_t + alignment_shift_adjustment

            if adjust_shift:
                # Apply the median shift between the target and source cells
                coords_shift = group_shift(target_cell_pc_feature[idx_ctype_t,:], source_cell_pc_feature, coords_target[idx_ctype_t,:], coords_source, working_memory_t = working_memory_t, metric_t = metric_t)
                coords_source_t = coords_source + coords_shift
                print(coords_shift)
            else:
                coords_source_t = coords_source.copy()

            project_ind[idx_ctype_t,:],project_weight[idx_ctype_t,:],cdists[idx_ctype_t,:],physical_dist[idx_ctype_t,:],all_avg_feat[idx_ctype_t,:] = physical_dist_priority_project(
                feat_target = target_cell_pc_feature[idx_ctype_t,:],
                feat_source = source_cell_pc_feature,
                coords_target = coords_target[idx_ctype_t,:],
                coords_source = coords_source_t,
                source_feat = source_feat,
                k2 = 1,
                pdist_thres = dist_thres,
                metric_t = metric_t,
                working_memory_t = working_memory_t)
            
    else:
        # if no source_sample_ctype_col is given, run the projection for all cells
        ave_dist_t,_,_,_ = average_dist(coords_target.copy(),working_memory_t=working_memory_t,strategy_t='delaunay')
        dist_thres = ave_dist_fold * ave_dist_t + alignment_shift_adjustment

        project_ind,project_weight,cdists,physical_dist,all_avg_feat = physical_dist_priority_project(
                feat_target = target_cell_pc_feature,
                feat_source = source_cell_pc_feature,
                coords_target = coords_target,
                coords_source = coords_source,
                source_feat = source_feat,
                k2 = 1,
                pdist_thres = dist_thres,
                working_memory_t = working_memory_t)


    umap_target = sdata_inte[idx_target,:].obsm[umap_feature]
    umap_source = sdata_inte[idx_source,:].obsm[umap_feature]

    # add the normalized projected data to the anndata object
    sdata_ref.layers[f'{source_sample}_raw'] = csr(all_avg_feat)
    sdata_ref.layers[f'{target_sample}_norm1e4'] = csr(sc.pp.normalize_total(sdata_ref,target_sum=1e4,layer = f'{raw_layer}',inplace=False)['X'])
    sdata_ref.layers[f'{source_sample}_norm1e4'] = csr(sc.pp.normalize_total(sdata_ref,target_sum=1e4,layer = f'{source_sample}_raw',inplace=False)['X'])
    
    # get the true and predicted cell type labels
    y_true_t = np.array(sdata_inte[idx_target].obs[source_sample_ctype_col].values) if source_sample_ctype_col is not None else None
    y_source = np.array(sdata_inte[idx_source].obs[source_sample_ctype_col].values) if source_sample_ctype_col is not None else None
    y_pred_t = y_source[project_ind[:,0]] if source_sample_ctype_col is not None else None

    torch.save([physical_dist,project_ind,coords_target,coords_source,y_true_t,y_pred_t,y_source,output_path,source_sample_ctype_col,umap_target,umap_source,source_sample,target_sample,cdists,k2],f'{output_path}/mid_result{batch_t}.pt')
    
    # generate evaluation plots
    if ifplot == True:
        evaluation_project(
            physical_dist = physical_dist,
            project_ind = project_ind,
            coords_target = coords_target,
            coords_source = coords_source,
            y_true_t = y_true_t,
            y_pred_t = y_pred_t,
            y_source = y_source,
            output_path = output_path,
            source_sample_ctype_col = source_sample_ctype_col,
            umap_target = umap_target,
            umap_source = umap_source,
            source_sample = source_sample,
            target_sample = target_sample,
            cdists = cdists,
            batch_t = batch_t,
            color_dict = color_dict)
        
    return sdata_ref,[project_ind,project_weight,cdists,physical_dist]

def average_dist(coords,quantile_t = 0.99,working_memory_t = 1000,strategy_t = 'convex'):
    """
    Finds the average distance between the cells after filtering for distances in the top quantile_t.
    
    Parameters
    ----------
    coords : array-like 
        The coordinates of the cells.
    quantile_t : float, optional (default: 0.99)
        The quantile to filter the delaunay graph edges. This is not applied if the number of cells is less than 5.
    working_memory_t : int, optional (default: 1000)
        The sought maximum memory for the chunked pairwise distance calculations.
    strategy_t : 'convex' | 'delaunay', optional (default: 'convex')
        The strategy to use for generating the delaunay graph.\n
        Convex will use Veronoi polygons clipped to the convex hull of the points and their rook spatial weights matrix (with libpysal).\n
        Delaunay will use the Delaunay triangulation (with sciipy).
    
    Returns
    -------
    float, float, np.ndarray, np.ndarray
        The average distance, the quantile_t of the edge distances, the edge distances, and the delaunay graph. On a dataset of less than 5 cells, the average distance is calculated directly and the other return values are the empty string.
    """

    coords_t = pd.DataFrame(coords)
    coords_t.drop_duplicates(inplace = True)
    coords = np.array(coords_t)

    if coords.shape[0] > 5:
        # on a larger dataset, calculate the pairwise distances in chunks and filter the edges in the delaunay graph

        delaunay_graph_t = coords2adjacentmat(coords,output_mode='raw',strategy_t = strategy_t)
        edges = np.array(delaunay_graph_t.edges())

        # caluclate the pairwise distances in chunks
        def reduce_func(chunk_t, start):
            return chunk_t
        dists = pairwise_distances_chunked(coords, coords, metric='euclidean', n_jobs=-1,working_memory = working_memory_t,reduce_func = reduce_func)
        
        # combine the edge distances into one list  
        edge_dist = []
        start_t = 0
        for dist_mat_t in dists:
            end_t = start_t + dist_mat_t.shape[0]
            idx_chunk = (start_t <= edges[:,0]) & (edges[:,0] < end_t)
            edge_t = edges[idx_chunk,:]
            edge_dist_t = np.array([dist_mat_t[node - start_t,val] for [node,val] in edge_t])
            edge_dist.extend(edge_dist_t)
            start_t = end_t

        # filter the edges in the delaunay graph by the quantile
        filter_thres = np.quantile(edge_dist,quantile_t)
        for i,j in edges[edge_dist > filter_thres,:]:
            delaunay_graph_t.remove_edge(i,j)

        # take the average of the remaining cells 
        result_t = np.mean(np.array(edge_dist)[edge_dist <= filter_thres])
        return result_t,filter_thres,edge_dist,delaunay_graph_t
    
    else:
        # on a small dataset, directly caluclate the pairwise distances and return the average 
        dists = pairwise_distances(coords, coords, metric='euclidean', n_jobs=-1)
        result_t = np.mean(dists.flatten())
        return result_t,'','',''

def group_shift(feat_target, feat_source, coords_target_t, coords_source_t, working_memory_t = 1000, pencentile_t = 0.8, metric_t = 'cosine'):
    """
    Calculates the median shift between the target and source cells.

    Parameters
    ----------
    feat_target : array-like
        The target features, used to calculate the pairwise distance between target and source cells.
    feat_source : array-like
        The source features, used to calculate the pairwise distance between target and source cells.
    coords_target_t : np.ndarray
        The coordinates of the target cells.
    coords_source_t : np.ndarray
        The coordinates of the source cells.
    working_memory_t : int, optional (default: 1000)
        The sought maximum memory for the chunked pairwise distance calculations.
    pencentile_t : float, optional (default: 0.8)
        The pencentile of the pairwise distances to use as anchor points.
    metric_t : str, optional (default: 'cosine')
        The metric to use for the pairwise distance calculations. See sklearn.metrics.pairwise_distances_chunked for more information. 
    
    Returns
    -------
    np.ndarray
        The median shift between the target and source cells.
    """

    # calulcate the pairwise distances between the target and source cells
    from sklearn.metrics import pairwise_distances_chunked
    print(f'Using {metric_t} distance to calculate group shift:')
    feat_similarity_ctype = np.vstack(list(pairwise_distances_chunked(feat_target, feat_source, metric=metric_t, n_jobs=-1, working_memory=working_memory_t)))
    
    # filter the top pencentile_t of the pairwise distances and use those as anchor points
    num_anchor = int(feat_similarity_ctype.shape[0] * pencentile_t)
    anchor_rank = np.argpartition(feat_similarity_ctype, num_anchor - 1, axis=-1)[:,:num_anchor]
    anchors = []
    for i in range(num_anchor):
        anchors.extend(anchor_rank[:,i].tolist())
        anchors = list(set(anchors))
        if len(anchors) >= num_anchor:
            break

    # calculate the median shift between the target cells and the anchor source cells 
    coords_shift = np.median(coords_target_t,axis=0) - np.median(coords_source_t[np.array(anchors),:],axis=0)
    return coords_shift

def physical_dist_priority_project(feat_target, feat_source, coords_target, coords_source, source_feat = None, k2 = 1, k_extend = 20, pdist_thres = 200, working_memory_t = 1000, metric_t = 'cosine'):
    """
    Gets the indicies, weights, and distances of the k-nearest neighbors for each cell in the target space.

    Parameters
    ----------
    feat_target : array-like
        The target features, used to calculate the pairwise distance between target and source cells.
    feat_source : array-like
        The source features, used to calculate the pairwise distance between target and source cells.
    coords_target : array-like
        The coordinates of the target cells.
    coords_source : array-like
        The coordinates of the source cells.
    source_feat : scipy.sparse matrix, optional
        If provided, also returns the weighted average of these source features.
    k2 : int, optional (default: 1)
        For points without `k2` neighbors within the physical distance threshold, extend the search to `k_extend` neighbors.
    k_extend : int, optional (default: 20)
        For points without `k2` neighbors within the physical distance threshold, extend the search to `k_extend` neighbors.
    pdist_thres : int, optional (default: 200)
        The physical distance threshold for the nearest neighbors search.
    working_memory_t : int, optional (default: 1000)
        The sought maximum memory for the chunked pairwise distance calculations.
    metric_t : str, optional (default: 'cosine')
        The metric to use for the pairwise distance calculations. See sklearn.metrics.pairwise_distances_chunked for more information. 

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray (, np.ndarray)
        The indicies of the k-nearest neighbors, their corresponding weights, cosine distances and physical distances for each cell.
        If `source_feat` is provided, also return the weighted average of the source features.
    """

    def reduce_func_cdist_priority(chunk_cdist, start):
        """
        Calculate the cosine distance and weights for each target cell to its k2 nearest neighbors in the chunk.

        Parameters
        ----------
        chunk_cdist : array-like (in CAST Stack, part of anndata.obsm)
            A continuous vertical slide of the pairwise distance matrix.
        start : int
            The starting index for this chunk.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            The indicies of the k-nearest neighbors, their corresponding weights, cosine distances and physical distances for each cell in the chunk.
        """

        # Calculate pairwise distance between target and source cells
        chunk_pdist = pairwise_distances(coords_target[start:(chunk_cdist.shape[0] + start),:],coords_source, metric='euclidean', n_jobs=-1)
        
        # Threshold the distance to nearby points by pdist_thres and get the number of nearby points
        idx_pdist_t = chunk_pdist < pdist_thres
        idx_pdist_sum = idx_pdist_t.sum(1)
        idx_lessk2 = (idx_pdist_sum>= k2)

        # Initialize the output arrays
        cosine_knn_ind = np.zeros([chunk_cdist.shape[0],k2]).astype(int)
        cosine_knn_weight = np.zeros_like(cosine_knn_ind).astype(float)
        cosine_knn_cdist = np.zeros_like(cosine_knn_ind).astype(float)
        cosine_knn_physical_dist = np.zeros_like(cosine_knn_ind).astype(float)
        
        idx_narrow = np.where(idx_lessk2)[0]
        idx_narrow_reverse = np.where(np.logical_not(idx_lessk2))[0]

        # for points with k2 neighbors, calculate their cosine distance and weights
        for i in idx_narrow:

            # filter for only the k2 nearest neighbors
            idx_pdist_t_i = idx_pdist_t[i,:]
            idx_i = np.where(idx_pdist_t[i,:])[0]
            knn_ind_t = idx_i[np.argpartition(chunk_cdist[i,idx_pdist_t_i], k2 - 1, axis=-1)[:k2]]

            # calculate cosine distance and weights
            _,weight_cell,cdist_cosine = cosine_IDW(chunk_cdist[i,knn_ind_t],k2 = k2,need_filter=False)
            cosine_knn_ind[[i],:] = knn_ind_t
            cosine_knn_weight[[i],:] = weight_cell
            cosine_knn_cdist[[i],:] = cdist_cosine
            cosine_knn_physical_dist[[i],:] = chunk_pdist[i,knn_ind_t]

        # for points without k2 neighbors, extend the search to k_extend neighbors
        if len(idx_narrow_reverse) > 0:
            for i in idx_narrow_reverse:
                idx_pdist_extend = np.argpartition(chunk_pdist[i,:], k_extend - 1, axis=-1)[:k_extend]

                # identical processing as above for the extended neighbors
                knn_ind_t = idx_pdist_extend[np.argpartition(chunk_cdist[i,idx_pdist_extend], k2 - 1, axis=-1)[:k2]]
                _,weight_cell,cdist_cosine = cosine_IDW(chunk_cdist[i,knn_ind_t],k2 = k2,need_filter=False)
                cosine_knn_ind[[i],:] = knn_ind_t
                cosine_knn_weight[[i],:] = weight_cell
                cosine_knn_cdist[[i],:] = cdist_cosine
                cosine_knn_physical_dist[[i],:] = chunk_pdist[i,knn_ind_t]

        return cosine_knn_ind,cosine_knn_weight,cosine_knn_cdist,cosine_knn_physical_dist
    
    print(f'Using {metric_t} distance to calculate cell low dimensional distance:')
    dists = pairwise_distances_chunked(feat_target, feat_source, metric=metric_t, n_jobs=-1,working_memory = working_memory_t,reduce_func=reduce_func_cdist_priority)
    cosine_knn_inds = []
    cosine_k2nn_weights = []
    cosine_k2nn_cdists = []
    cosine_k2nn_physical_dists = []
    for output in tqdm(dists):
        cosine_knn_inds.append(output[0])
        cosine_k2nn_weights.append(output[1])
        cosine_k2nn_cdists.append(output[2])
        cosine_k2nn_physical_dists.append(output[3])

    all_cosine_knn_inds = np.concatenate(cosine_knn_inds)
    all_cosine_k2nn_weights = np.concatenate(cosine_k2nn_weights)
    all_cosine_k2nn_cdists = np.concatenate(cosine_k2nn_cdists)
    all_cosine_k2nn_physical_dists = np.concatenate(cosine_k2nn_physical_dists)
    
    # if source features are provided, also take the weighted average of the source features
    if source_feat is not None:
        mask_idw = sparse_mask(all_cosine_k2nn_weights,all_cosine_knn_inds, source_feat.shape[0])
        all_avg_feat = mask_idw.dot(source_feat)
        return all_cosine_knn_inds,all_cosine_k2nn_weights,all_cosine_k2nn_cdists,all_cosine_k2nn_physical_dists,all_avg_feat
    
    else:
        return all_cosine_knn_inds,all_cosine_k2nn_weights,all_cosine_k2nn_cdists,all_cosine_k2nn_physical_dists


def sparse_mask(idw_t, ind : np.ndarray, n_cols : int, dtype=np.float64): # ind is indices with shape (num data points, indices), in the form of output of numpy.argpartition function
    """
    Creates a CSR matrix from the given non-zero values and their corresponding indices.

    Parameters
    ----------
    idw_t : np.ndarray
        The non-zero values to set in the matrix.
    ind : np.ndarray
        The indices of the non-zero values with shape (num data points, indices), in the format of the output of the `numpy.argpartition` function
    n_cols : int
        The number of columns in the output matrix.
    dtype : type, optional (default: np.float64)
        The data type of the output matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The CSR matrix with the given non-zero values and indices.
    """

    # build csr matrix from scratch
    rows = np.repeat(np.arange(ind.shape[0]), ind.shape[1]) # gives like [1,1,1,2,2,2,3,3,3]
    cols = ind.flatten() # the col indices that should be 1
    data = idw_t.flatten() # Set to `1` each (row,column) pair
    return csr_matrix((data, (rows, cols)), shape=(ind.shape[0], n_cols), dtype=dtype)

def cosine_IDW(cosine_dist_t,k2=5,eps = 1e-6,need_filter = True,ifavg = False):
    """
    Compute the weights for the k-nearest neighbors of a target cell using the inverse distance weighting method or the average weight.

    Parameters
    ----------
    cosine_dist_t : np.ndarray
        The cosine distance between the target cell and its neighbors.
    k2 : int, optional (default: 5)
        The number of nearest neighbors to consider.
    eps : float, optional (default: 1e-6)
        A small constant to prevent dividing by zero.
    need_filter : bool, optional (default: True)
        Whether to filter for only the k-nearest neighbors. If True, only consider the `k2` nearest neighbors.
    ifavg : bool, optional (default: False)
        Whether to use the average weight for all the neighbors. If False, use the IDW method.

    Returns
    -------
    np.ndarray
        The indicies of the `k2`-nearest neighbors (in `cosine_dist_t`) if `need_filter` is True, otherwise 0.
    np.ndarray
        The cell weights as a 1D array. If `ifavg` is True, a uniform `1/k2` weight for `k2` neighbors. Otherwise the IDW weights for the `k2` neighbors if `need_filter` is True or for all cells if `need_filter` is False.
    np.ndarray
        The cosine distances for the `k2`-nearest neighbors if `need_filter` is True, otherwise `cosine_dist_t`. 
    """

    if need_filter:
        idx_cosdist_t = np.argpartition(cosine_dist_t, k2 - 1, axis=-1)[:k2]
        cdist_cosine_t = cosine_dist_t[idx_cosdist_t]
    else:
        idx_cosdist_t = 0
        cdist_cosine_t = cosine_dist_t
    if ifavg:
        weight_cell_t = np.array([1/k2] * k2)
    else:
        weight_cell_t = IDW(cdist_cosine_t,eps)
    return idx_cosdist_t, weight_cell_t, cdist_cosine_t

def IDW(df_value,eps = 1e-6):
    """
    Calculates the normalized, reciprocal weights for an array.

    Parameters
    ----------
    df_value : np.ndarray
        The array to take the weights from.
    eps : float, optional (default: 1e-6)
        A small constant to prevent dividing by zero.
    
    Returns
    -------
    np.ndarray
        The normalized, reciprocal weights for each element.
    """

    weights = 1.0 /(df_value + eps).T # invert the weights 
    weights /= weights.sum(axis=0) # normalize the weights
    return weights.T

def evaluation_project(physical_dist, project_ind, coords_target, coords_source,
    y_true_t, y_pred_t, y_source, output_path, source_sample_ctype_col,
    umap_target = None, umap_source = None, source_sample = None, target_sample = None,
    cdists = None, batch_t = '', exclude_group = 'Other', color_dict = None, umap_examples = False):
    """
    Generates and saves evaluation plots for the projection results, such as the physical distance and cosine distance histograms, the confusion matrix, a 3D link plot, and UMAP examples plot (see `cdist_check`).

    Parameters
    ----------
    physical_dist : array-like
        The physical distances between the target cells and their k-nearest neighbors.
    project_ind : array-like
        The indicies of the k-nearest neighbors for each cell in the target space.
    coords_target : array-like
        The coordinates of the target cells.
    coords_source : array-like
        The coordinates of the source cells.
    y_true_t : np.ndarray
        The true cell type labels of the target cells.
    y_pred_t : np.ndarray
        The predicted cell type labels of the target cells based on the projection results.
    y_source : array-like
        The cell type labels of the source cells.
    output_path : str
        The path to save the output files.
    source_sample_ctype_col : str
        The column name of the cell type annotation in the source sample. If omitted, no confusion matrix will be plotted and visualizations won't include cell type information. 
    umap_target : array-like, optional
        The UMAP coordinates of the target cells.
    umap_source : array-like, optional
        The UMAP coordinates of the source cells.
    source_sample : str, optional
        The name of the source sample, used as a label on the UMAP examples plot (if `umap_examples` is True).
    target_sample : str, optional
        The name of the target sample, used as a label on the UMAP examples plot (if `umap_examples` is True).
    cdists : array-like, optional
        The cosine distances between the target cells and their k-nearest neighbors.
    batch_t : str, optional (default: '')
        The batch name for naming the output files.
    exclude_group : str | None, optional (default: 'Other')
        The group to exclude from the confusion matrix. If None, exclude no groups. 
    color_dict : dict, optional
        The color dictionary for visualizing the cell type annotations. (only applied if `source_sample_ctype_col` is given).
    umap_examples : bool, optional (default: False)
        Whether to generate the UMAP examples plot.
    """

    print(f'Generate evaluation plots:')
    plt.rcParams.update({'pdf.fonttype':42, 'font.size' : 15})
    plt.rcParams['axes.grid'] = False

    ### histograms ###
    cdist_hist(physical_dist.flatten(),range_t = [0,2000]) # Physical distance 
    plt.savefig(f'{output_path}/physical_dist_hist{batch_t}.pdf')
    cdist_hist(cdists.flatten(),range_t = [0,2]) # Cosine distance
    plt.savefig(f'{output_path}/cdist_hist{batch_t}.pdf')


    ### confusion matrix ###
    if source_sample_ctype_col is not None:
        if exclude_group is not None:
            idx_t = y_true_t != exclude_group
            y_true_t_use = y_true_t[idx_t]
            y_pred_t_use = y_pred_t[idx_t]
        else:
            y_true_t_use = y_true_t
            y_pred_t_use = y_pred_t

        confusion_mat_plot(y_true_t_use,y_pred_t_use) # with label 
        plt.savefig(f'{output_path}/confusion_mat_raw_with_label_{source_sample_ctype_col}{batch_t}.pdf')
        confusion_mat_plot(y_true_t_use,y_pred_t_use,withlabel = False) # without label 
        plt.savefig(f'{output_path}/confusion_mat_raw_without_label_{source_sample_ctype_col}{batch_t}.pdf')


    ### link plot 3d ###
    if color_dict is not None and source_sample_ctype_col is not None:
        color_target = [color_dict[x] for x in y_true_t]
        color_source = [color_dict[x] for x in y_source]
    else:
        color_target="#9295CA" 
        color_source='#E66665'

    link_plot_3d(project_ind, coords_target, coords_source, k = 1,figsize_t = [10,10], 
                sample_n=200, link_color_mask = None, 
                color_target = color_target, color_source = color_source,
                color_true = "#222222")
    plt.savefig(f'{output_path}/link_plot{batch_t}.pdf', dpi=300)


    ### Umap ###
    if umap_examples:
        cdist_check(cdists.copy(),project_ind.copy(),umap_target,umap_source,labels_t=[target_sample,source_sample],random_seed_t=0,figsize_t=[40,32])
        plt.savefig(f'{output_path}/umap_examples{batch_t}.pdf',dpi = 300)

#################### Visualization ####################

def cdist_hist(data_t,range_t = None,step = None):
    """
    Generates a histogram of the given data.

    Parameters
    ----------
    data_t : array-like
        The data to plot.
    range_t : tuple, optional
        The range of the x-axis (inclusive on both sides). If omitted, the entire data range is included.
    step : float, optional
        The step size of the x-axis. If omitted, the step size is automatically determined by matplotlib.hist.
    """

    plt.figure(figsize=[5,5])
    plt.hist(data_t, bins='auto',alpha = 0.5,color = '#1073BC')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    if type(range_t) != type(None):
        if type(step) != type(None):
            plt.xticks(np.arange(range_t[0], range_t[1] + 0.001, step),fontsize=20) # +0.001 to include the last value
        else:
            plt.xticks(fontsize=20)
            plt.xlim(range_t[0], range_t[1])
    else:
        plt.xticks(fontsize=20)

    plt.tight_layout()

def confusion_mat_plot(y_true_t, y_pred_t, filter_thres = None, withlabel = True, fig_x = 60, fig_y = 20):
    """
    Generates a confusion matrix plot.
    
    Parameters
    ----------
    y_true_t : np.ndarray
        The true cell type labels.
    y_pred_t : np.ndarray
        The predicted cell type labels from the projection results.
    filter_thres : int, optional
        A threshold to filter out cell types with low cell counts.
    withlabel : bool, optional (default: True)
        Whether to include value labels on the diagonal of the confusion matrix.
    fig_x : int, optional (default: 60)
        The width of the figure.
    fig_y : int, optional (default: 20)
        The height of the figure.
    """

    plt.rcParams.update({'axes.labelsize' : 30,'pdf.fonttype':42,'axes.titlesize' : 30,'font.size': 15,'legend.markerscale' : 3})
    plt.rcParams['axes.grid'] = False
    TPrate = np.round(np.sum(y_pred_t == y_true_t) / len(y_true_t),2)
    uniq_t = np.unique(y_true_t,return_counts=True)

    # filter the cell types by their counts if filter_thres is given
    if type(filter_thres) == type(None):
        labels_t = uniq_t[0]
    else:
        labels_t = uniq_t[0][uniq_t[1] >= filter_thres]

    plt.figure(figsize=[fig_x,fig_y])

    for idx_t, i in enumerate(['count','true','pred']):
        if i == 'count':
            normalize_t = None
            title_t = 'Counts (TP%%: %.2f)' % TPrate
        elif i == 'true':
            normalize_t = 'true'
            title_t = 'Sensitivity'
        elif i == 'pred':
            normalize_t = 'pred'
            title_t = 'Precision'
        plt.subplot(1,3,idx_t + 1)
        
        confusion_mat = confusion_matrix(y_true_t,y_pred_t,labels = labels_t, normalize = normalize_t)
        if i == 'count':
            vmax_t = np.max(confusion_mat)
        else:
            vmax_t = 1
        confusion_mat = pd.DataFrame(confusion_mat,columns=labels_t,index=labels_t)

        # add labels 
        if withlabel:
            annot = np.diag(np.diag(confusion_mat.values.copy(),0),0)
            annot = np.round(annot,2)
            annot = annot.astype('str')
            annot[annot=='0.0']=''
            annot[annot=='0']=''
            sns.heatmap(confusion_mat,cmap = 'RdBu',center = 0,annot=annot,fmt='',square = True,vmax = vmax_t)
        else:
            sns.heatmap(confusion_mat,cmap = 'RdBu',center = 0,square = True,vmax = vmax_t)

        plt.title(title_t)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

def cdist_check(cdist_t,cdist_idx,umap_coords0,umap_coords1, labels_t = ['query','ref'],random_seed_t = 2,figsize_t = [40,32],output_path_t = None):
    """
    Generates UMAP examples plots — for a random 20 points (generating 20 subplots), highlights the target point and its nearest neighbor in the reference sample.

    Parameters
    ----------
    cdist_t : array-like
        The cosine distances between the target cells and their k-nearest neighbors (used to title the subplots with their distance values).
    cdist_idx : array-like
        The indicies of the k-nearest neighbors for each cell in the target space.
    umap_coords0 : array-like
        The UMAP coordinates of the query cells.
    umap_coords1 : array-like
        The UMAP coordinates of the reference cells.
    labels_t : list[str], optional (default: ['query','ref'])
        The labels for the query and reference samples.
    random_seed_t : int, optional (default: 2)
        The random seed used to sample the 20 random points.
    figsize_t : list[float], optional (default: [40,32])
        The size of the figure.
    output_path_t : str, optional
        The path to save the final plot. If omitted, the plot will not be saved.
    """

    plt.rcParams.update({'xtick.labelsize' : 20,'ytick.labelsize':20, 'axes.labelsize' : 30, 'axes.titlesize' : 40,'axes.grid': False})
    random.seed(random_seed_t)

    # randomly sample 20 points to plot
    sampled_points = np.sort(random.sample(list(range(0,cdist_idx.shape[0])),20))
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=figsize_t)
    axs = axs.flatten()

    for i in range(len(sampled_points)): # for each point 
        idx_check = sampled_points[i]

        # plot all points in the target and source samples
        axs[i].scatter(umap_coords0[:,0],umap_coords0[:,1],s = 0.5,c = '#1f77b4',rasterized=True) # query in blue
        axs[i].scatter(umap_coords1[:,0],umap_coords1[:,1],s = 0.5,c = '#E47E8B',rasterized=True) # reference in pink

        # highlight the target cell and its nearest neighbor in the source sample
        axs[i].scatter(umap_coords0[idx_check,0],umap_coords0[idx_check,1],s = 220,linewidth = 4,c = '#1f77b4',edgecolors = '#000000',label = labels_t[0],rasterized=False)
        axs[i].scatter(umap_coords1[cdist_idx[idx_check,0],0],umap_coords1[cdist_idx[idx_check,0],1],s = 220,linewidth=4,c = '#E47E8B',edgecolors = '#000000',label = labels_t[1], rasterized=False)
        
        axs[i].legend(scatterpoints=1,markerscale=2, fontsize=30)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title('cdist = ' + str(format(cdist_t[idx_check,0],'.2f')))
    
    # save the output
    if output_path_t is not None:
        plt.savefig(f'{output_path_t}/umap_examples.pdf',dpi = 300)
        plt.close('all')


def link_plot_3d(assign_mat, coords_target, coords_source, k, figsize_t = [15,20], sample_n=1000, link_color_mask=None, color_target="#9295CA", color_source='#E66665', color_true = "#999999", color_false = "#999999", remove_background = True):
    """
    Generates a 3D link plot for the projection results - the target cells are displayed in a plane above the source cells and `sample_n` links between corresponding target and source cells are drawn.

    Parameters
    ----------
    assign_mat : array-like
        The indicies of the k-nearest neighbors for each cell in the target space.
    coords_target : array-like
        The coordinates of the target cells.
    coords_source : array-like
        The coordinates of the source cells.
    k : int
        The number of nearest neighbors to consider. As of the current implementation, this must be 1. 
    figsize_t : list, optional (default: [15,20])
        The size of the figure.
    sample_n : int, optional (default: 1000)
        The number of links to sample and display.
    link_color_mask : np.ndarray, optional
        A boolean mask to color the links based on their corresponding values. If omitted, all links will be colored with `color_true`.
    color_target : str, optional (default: "#9295CA")
        The color of the target cells (displayed in a plane above the source cells).
    color_source : str, optional (default: '#E66665')
        The color of the source cells (displayed in a plane below the target cells).
    color_true : str, optional (default: "#999999") 
        The color of the links that are True in the `link_color_mask`. If `link_color_mask` is omitted, this color will be used for all links.
    color_false : str, optional (default: "#999999")
        The color of the links that are False in the `link_color_mask`. If `link_color_mask` is omitted, this color will not be used.
    remove_background : bool, optional (default: True)
        Whether to remove the axes of the plot.
    """

    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    assert k == 1

    # initialize the plot 
    ax = plt.figure(figsize=figsize_t).add_subplot(projection='3d')
    xylim = max(coords_source.max(), coords_target.max())
    ax.set_xlim(0, xylim)
    ax.set_ylim(0, xylim)
    ax.set_zlim(-0.1, 1.1)
    ax.set_box_aspect([1,1,0.6])
    ax.view_init(elev=25)

    # get the coordinates of the source cells that are linked to the target cells
    coordsidx_transfer_source_link = assign_mat[:, 0]
    
    coords_transfer_source_link = coords_source[coordsidx_transfer_source_link,:]
    t1 = np.row_stack((coords_transfer_source_link[:,0],coords_transfer_source_link[:,1])) # source
    t2 = np.row_stack((coords_target[:,0],coords_target[:,1])) # target
    
    downsample_indices = np.random.choice(range(coords_target.shape[0]), sample_n)
    
    if link_color_mask is not None:
        # draw the links with different colors based on the link_color_mask (with color_false)
        final_true_indices = np.intersect1d(downsample_indices, np.where(link_color_mask)[0])
        final_false_indices = np.intersect1d(downsample_indices, np.where(~link_color_mask)[0])
        segs = [[(*t2[:, i], 0), (*t1[:, i], 1)] for i in final_false_indices]
        line_collection = Line3DCollection(segs, colors=color_false, lw=0.5, linestyles='dashed')
        line_collection.set_rasterized(True)
        ax.add_collection(line_collection)
    else:
        final_true_indices = downsample_indices
        
    # draw the links with color_true
    segs = [[(*t2[:, i], 0), (*t1[:, i], 1)] for i in final_true_indices]
    line_collection = Line3DCollection(segs, colors=color_true, lw=0.5, linestyles='dashed')
    line_collection.set_rasterized(True)
    ax.add_collection(line_collection)
    
    # plot the target at plane z = 0
    ax.scatter(xs = coords_target[:,0],ys = coords_target[:,1], zs=0, s = 2, c =color_target, alpha = 0.8, ec='none', rasterized=True, depthshade=False)
    # plot the source at plane z = 1
    ax.scatter(xs = coords_source[:,0],ys = coords_source[:,1], zs=1, s = 2, c =color_source, alpha = 0.8, ec='none', rasterized=True, depthshade=False)
    
    # Remove background
    if remove_background:
        # Remove axis
        ax.axis('off')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
