import networkx as nx
import numpy as np
import pandas as pd
import scipy, random
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances
from visualize import link_plot

def coords2adjacentmat(coords,output_mode = 'adjacent',strategy_t = 'convex'):
    """
    Given a spatial data matrix, generate the delaunay graph in the specified format.

    Parameters
    ----------
    coords : ndarray
        The spatial data matrix with the coordinate position for each cell.
    output_mode : 'adjacent' | 'raw' | 'adjacent_sparse', optional (default: 'adjacent')
        The output format of the delaunay graph.
        If 'adjacent', the function will return the adjacent matrix.
        If 'raw', the function will return the raw delaunay graph.
        If 'adjacent_sparse', the function will return the adjacent matrix in sparse format.
    strategy_t : 'convex' | 'delaunay', optional (default: 'convex')
        The strategy to use for generating the delaunay graph.
        Convex will use Veronoi polygons clipped to the convex hull of the points and their rook spatial weights matrix (with libpysal).
        Delaunay will use the Delaunay triangulation (with sciipy).
    
    Returns
    -------
    delaunay_graph : ndarray | nx.Graph | scipy.sparse.csr_matrix
        The delaunay graph in the specified format.

    """


    if strategy_t == 'convex': ### slow but may generate more reasonable delaunay graph
        # Generate the Delaunay graph using Veronoi polygons clipped to the convex hull of the points and their rook spatial weights matrix 
        from libpysal.cg import voronoi_frames
        from libpysal import weights
        cells, _ = voronoi_frames(coords, clip="convex hull")
        delaunay_graph = weights.Rook.from_dataframe(cells).to_networkx()

    elif strategy_t == 'delaunay': ### fast but may generate long distance edges
        # Generate the Delaunay graph using the Delaunay triangulation
        from scipy.spatial import Delaunay
        from collections import defaultdict
        tri = Delaunay(coords)
        delaunay_graph = nx.Graph()

        # create a dictionary of coordinates to their indices
        coords_dict = defaultdict(list)
        for i, coord in enumerate(coords):
            coords_dict[tuple(coord)].append(i)

        # put all of the calculated edges into the graph
        for simplex in tri.simplices:
            for i in range(3):
                for node1 in coords_dict[tuple(coords[simplex[i]])]:
                    for node2 in coords_dict[tuple(coords[simplex[(i+1)%3]])]:
                        if not delaunay_graph.has_edge(node1, node2):
                            delaunay_graph.add_edge(node1, node2)

    # Return the output in the specified format (adjacent, raw, adjacent_sparse)
    if output_mode == 'adjacent':
        return nx.to_scipy_sparse_array(delaunay_graph).todense()
    elif output_mode == 'raw':
        return delaunay_graph
    elif output_mode == 'adjacent_sparse':
        return nx.to_scipy_sparse_array(delaunay_graph)

def hv_cutoff(max_col,threshold_cell_num=2000):
    """
    Returns the highest integer threshold such that the at least threshold_cell_num cells have a max_col value greater than the threshold.

    Parameters
    ----------
    max_col : np.ndarray
        The data to threshold.
    threshold_cell_num : int, optional (default: 2000)
        The threshold such that at least threshold_cell_num cells have a max_col value greater than the threshold.

    Returns
    -------
    thres_t : int
        The highest integer threshold such that the at least threshold_cell_num cells have a max_col value greater than the threshold. 
    """
    for thres_t in range(0,int(np.max(max_col))):
        if np.sum(max_col > thres_t) < threshold_cell_num:
            return thres_t -1

def detect_highly_variable_genes(sdata,batch_key = 'batch',n_top_genes = 4000,count_layer = 'count'):
    """
    Finds the genes that have high expression in at least one cell in all batches.

    Parameters
    ----------
    sdata : AnnData
        Annotated data matrix.
    batch_key : str, optional (default: 'batch')
        The column name of the samples in sdata.obs
    n_top_genes : int, optional (default: 4000)
        The number of genes to keep - the function result may contain more genes than this because it uses the smallest integer threshold to get at least this many genes.
    count_layer : str, optional (default: 'count')
        The layer in sdata.layers to use for the count data.
        If count_layer is '.X', the function will use sdata.X for the count data.

    Returns
    -------
    highly_variable_genes : np.ndarray
        A boolean array of which genes are highly variable across all batches.
    """

    samples = np.unique(sdata.obs[batch_key])
    thres_list = []
    max_count_list = []
    bool_list = []
    for list_t, sample_t in enumerate(samples):
        idx_t = sdata.obs[batch_key] == sample_t

        # max_t is the maximum expression of each gene in the batch across all cells
        if count_layer == '.X':
            max_t = sdata[idx_t,:].X.max(0).toarray() if scipy.sparse.issparse(sdata.X) else sdata[idx_t,:].X.max(0) #
        else:
            max_t = sdata[idx_t,:].layers[count_layer].max(0).toarray() if scipy.sparse.issparse(sdata.layers[count_layer]) else sdata[idx_t,:].layers[count_layer].max(0)
        
        max_count_list.append(max_t)
        thres_list.append(hv_cutoff(max_count_list[list_t],threshold=n_top_genes))
        
        # a boolean list describing whether the max expression of each gene is greater than the threshold (the highest integer threshold such that at least n_top_genes cells are kept)
        bool_list.append(max_count_list[list_t] > thres_list[list_t])

    # keep the genes that have max expression above the threshold in all batches
    stack = np.stack(bool_list)
    return np.all(stack, axis=0)[0]

def extract_coords_exp(sdata, batch_key = 'batch', cols = 'spatial', count_layer = 'count', data_format = 'norm1e4',ifcombat = False, if_inte = False):
    """
    Extracts the spatial data and gene expression data for each sample.

    Parameters
    ----------
    sdata : AnnData
        Annotated data matrix.
    batch_key : str, optional (default: 'batch')
        The column name of the samples in sdata.obs
    cols : 'spatial' | list, optional (default: 'spatial')
        The column name of the coordinates. If cols is 'spatial', the function will use sdata.obsm['spatial'] for the coordinates. Otherwise, the function will use sdata.obs[cols] for the coordinates.
    count_layer : str, optional (default: 'count')
        The layer in sdata.layers to use for the count data.
        If count_layer is '.X', the function will use sdata.X for the count data.
    data_format : str, optional (default: 'norm1e4')
        The name of the layer in sdata.layers to use for the data.
    ifcombat : bool, optional (default: False)
        Whether or not to use ComBat to correct for batch effects.
    if_inte : bool, optional (default: False)
        Whether or not to perform Harmony integration.
    
    Returns
    -------
    coords_raw : dict
        The spatial data matrix with the coordinate position for each cell, indexed by sample name.
    exps : dict
        The gene expression data for each coordinate in coords_raw, indexed by sample name.
    """

    coords_raw = {}
    exps = {}
    samples = np.unique(sdata.obs[batch_key])

    # convert the data to a csr and normalize the data
    if count_layer == '.X':
        sdata.layers['raw'] = sdata.X.copy()
    sdata = preprocess_fast(sdata, mode = 'customized')

    if if_inte:
        scaled_layer = 'log2_norm1e4_scaled'
        pc_feature = 'X_pca_harmony'
        sdata = Harmony_integration(sdata,
                                    scaled_layer = scaled_layer,
                                    use_highly_variable_t = True,
                                    batch_key = batch_key,
                                    umap_n_neighbors = 50,
                                    umap_n_pcs = 30,
                                    min_dist = 0.01,
                                    spread_t = 5,
                                    source_sample_ctype_col = None,
                                    output_path = None,
                                    n_components = 50,
                                    ifplot = False,
                                    ifcombat = ifcombat)
        for sample_t in samples:
            idx_t = sdata.obs[batch_key] == sample_t
            # extract the spatial data and gene expression data for each sample
            coords_raw[sample_t] = sdata.obsm['spatial'][idx_t] if type(cols) is not list else np.array(sdata.obs[cols][idx_t])
            exps[sample_t] = sdata[idx_t].obsm[pc_feature].copy()

    else:
        # perform ComBat if ifcombat (if if_inte, ComBat is performed in Harmony_integration) 
        sdata.X = sdata.layers[data_format].copy()
        if ifcombat == True:
            sc.pp.combat(sdata, key=batch_key)

        for sample_t in samples:
            idx_t = sdata.obs[batch_key] == sample_t
            # extract the spatial data and gene expression data for each sample
            coords_raw[sample_t] = sdata.obsm['spatial'][idx_t] if type(cols) is not list else np.array(sdata.obs[cols][idx_t])
            exps[sample_t] = sdata[idx_t].X.copy() 
            # if the data is sparse, convert it to a dense matrix
            if scipy.sparse.issparse(exps[sample_t]):
                exps[sample_t] = exps[sample_t].toarray()
                
    return coords_raw,exps

def Harmony_integration(sdata_inte, scaled_layer, use_highly_variable_t, batch_key,
    umap_n_neighbors, umap_n_pcs, min_dist, spread_t,
    source_sample_ctype_col, output_path, n_components = 50, ifplot = True, ifcombat = False):
    """
    Performs Harmony integration on the data.
    In addition, the function will run PCA and construct a neighborhood graph and UMAP with the integrated data.

    Parameters
    ----------
    sdata_inte : AnnData
        Annotated data matrix.
    scaled_layer : str
        The name of the layer in sdata.layers to use for the scaled data.
    use_highly_variable_t : bool
        Whether or not to use highly variable genes for the PCA.
    batch_key : str
        The column name of the samples in sdata.obs
    umap_n_neighbors : int
        The number of neighbors to use for the UMAP.
    umap_n_pcs : int
        The number of PCs to use for the UMAP.
    min_dist : float
        The minimum distance to use for the UMAP.
    spread_t : int
        The spread to use for the UMAP.
    source_sample_ctype_col : str
        The key in sdata.obs to split batches on.
    output_path : str
        The path to save the output.
    n_components : int (default: 50)
        The number of components to use for the Harmony integration.
    ifplot : bool (default: True)
        Whether or not to plot the UMAP.
    ifcombat : bool (default: False)
        Whether or not to use ComBat to correct for batch effects.
    
    Returns
    -------
    sdata_inte : AnnData
        The annotated data matrix, now supplemented with the neighborhood graph and UMAP information. 
    """

    sdata_inte.X = sdata_inte.layers[scaled_layer].copy()
    if ifcombat == True:
        sc.pp.combat(sdata_inte, key=batch_key)
        
    print(f'Running PCA based on the layer {scaled_layer}:')
    sc.tl.pca(sdata_inte, use_highly_variable=use_highly_variable_t, svd_solver = 'full', n_comps= n_components)
    print(f'Running Harmony integration:')
    sc.external.pp.harmony_integrate(sdata_inte, batch_key)
    print(f'Compute a neighborhood graph based on the {umap_n_neighbors} `n_neighbors`, {umap_n_pcs} `n_pcs`:')
    sc.pp.neighbors(sdata_inte, n_neighbors=umap_n_neighbors, n_pcs=umap_n_pcs, use_rep='X_pca_harmony')
    print(f'Generate the UMAP based on the {min_dist} `min_dist`, {spread_t} `spread`:')
    sc.tl.umap(sdata_inte,min_dist=min_dist, spread = spread_t)

    sdata_inte.obsm['har_X_umap'] = sdata_inte.obsm['X_umap'].copy()
    if ifplot == True:
        plt.rcParams.update({'pdf.fonttype':42})
        sc.settings.figdir = output_path
        sc.set_figure_params(figsize=(10, 10),facecolor='white',vector_friendly=True, dpi_save=300,fontsize = 25)
        sc.pl.umap(sdata_inte,color=[batch_key],size=10,save=f'_har_{umap_n_pcs}pcs_batch.pdf')
        sc.pl.umap(sdata_inte,color=[source_sample_ctype_col],size=10,save=f'_har_{umap_n_pcs}pcs_ctype.pdf') if source_sample_ctype_col is not None else None
    
    return sdata_inte

def random_sample(coords_t, nodenum, seed_t = 2):
    """
    Returns nodenum random indices from coords_t.

    Parameters
    ----------
    coords_t : ndarray | torch.Tensor
        The random indices will be sampled from the length of the first axis.
    nodenum : int
        The number of random indicies to return.
    seed_t : int | None, optional (default: '2')
        The seed for the random package.
    
    Returns
    -------
    sub_node_idx : np.ndarray
        1D nodenum-length sorted array of random indicies.
    """

    random.seed(seed_t)
    sub_node_idx = np.sort(random.sample(range(coords_t.shape[0]),nodenum))
    return sub_node_idx

def sub_node_sum(coords_t,exp_t,nodenum=1000,vis = True,seed_t = 2):
    """
    Randomly selects nodenum nodes from coords_t and returns their indices and subnode expression matrix for the nearest neighbor of each chosen point.

    Parameters
    ----------
    coords_t : ndarray
        The spatial data matrix with the coordinate position for each cell.
    exp_t : ndarray
        Gene expression data for each coordinate in coords_t.
    nodenum : int, optional (default: '1000')
        The number of nodes to return. 
        If this is larger than the total number of nodes, the original data will be returned.
    vis : bool, optional (default: 'True')
        Whether or not to visualize the subnodes and their neighbros before returning them. 
    seed_t : int, optional (default: '2')
        The seed for the random package (for reproducibility). 
    
    Returns
    -------
    exp_t_sub : ndarray
        The subnode expression matrix for the nearest neighbor of each chosen point.
    sub_node_idx : ndarray
        1D nodenum-length sorted array of random indicies from coords_t.
    """

    from scipy.sparse import csr_matrix as csr

    # if nodenum is larger than the total number of nodes, return the original data.
    if nodenum > coords_t.shape[0]:
        print('The number of nodes is larger than the total number of nodes. Return the original data.')
        sub_node_idx = np.arange(coords_t.shape[0]) # evenly spaced indices from 0 to the total number of nodes

        if scipy.sparse.issparse(exp_t):
            return exp_t,sub_node_idx 
        else:
            # if not sparse, return compressed sparse row matrix 
            return csr(exp_t),sub_node_idx 
        
    # gets nodenum random indices from coords_t and their corresponding coordinates
    sub_node_idx = random_sample(coords_t, nodenum, seed_t = seed_t)
    coords_t_sub = coords_t[sub_node_idx,:].copy()
    
    # get the nearest neighbors of the sub_node_idx in the original data
    close_idx = nearest_neighbors_idx(coords_t_sub,coords_t)

    # create a sparse matrix A where each row is a one-hot vector indicating the nearest neighbor
    A = np.zeros([coords_t_sub.shape[0],coords_t.shape[0]])
    for ind,i in enumerate(close_idx.tolist()):
        A[i,ind] = 1
    csr_A = csr(A)

    # multiply the one-hot matrix by the expression matrix to get the sub-node expression matrix
    if scipy.sparse.issparse(exp_t):
        exp_t_sub = csr_A.dot(exp_t)
    else:
        exp_t_sub = csr_A.dot(csr(exp_t))
    
    # if vis is True, plot the link plot
    if(vis == True):
        link_plot(close_idx,coords_t,coords_t_sub,k = 1)
    return exp_t_sub,sub_node_idx

def nearest_neighbors_idx(coord1,coord2,mode_t = 'knn'): ### coord1 is the reference, coord2 is the target
    """
    Find each coord2 point's the nearest neighbor in coord1.
    
    Parameters 
    ----------
    coord1 : array-like
        The reference array of coordinates. 
    coord2 : array-like
        The query array of coordinates.
    mode_t : str, optional (default: 'knn')
        Whether or not to use a KNN classifier.

    Returns
    -------
    close_idx : ndarray
        For each point in coord2, the index of its nearest neighbor in coord1.
    """

    if mode_t == 'knn':
        # Use a KNN classifier to find the nearest neighbor
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=1,metric='euclidean')

        knn_classifier.fit(coord1, np.zeros(coord1.shape[0]))  # Use dummy labels, since we only care about distances
        
        _, close_idx = knn_classifier.kneighbors(coord2)
        return close_idx
    
    else:
        # Use pairwise distances to find the nearest neighbor
        result = []
        dists = pairwise_distances_chunked(coord2,coord1,working_memory = 100, metric='euclidean', n_jobs=-1)
        for chunk in tqdm(dists): # for each chunk (minibatch)
            knn_ind = np.argpartition(chunk, 0, axis=-1)[:, 0] # introsort to get indices of top k neighbors according to the distance matrix [n_query, k]
            result.append(knn_ind)

        close_idx = np.concatenate(result)
        return np.expand_dims(close_idx, axis=1)
    
def non_zero_center_scale(sdata_t_X):
    """
    Scale the data by dividing each column by its standard deviation without centering (i.e. without subtracting the mean).

    Parameters 
    ----------
    sdata_t_X : ndarray
        The data matrix to scale.
    """

    # Calculate the column-wise standard deviation without centering (i.e. wihtoit subtracting the mean)
    std_nocenter = np.sqrt(np.square(sdata_t_X).sum(0)/(sdata_t_X.shape[0]-1))

    # Divide each column by the standard deviation
    return(sdata_t_X/std_nocenter)

def sub_data_extract(sample_list,coords_raw, exps, nodenum_t = 20000, if_non_zero_center_scale = True):
    """
    Extracts nodenum_t random nodes in sample_list and returns their coordinates and expression matrix for each sample.

    Parameters
    ----------
    sample_list : list
        The list of samples to extract sub-data from.
    coords_raw : dict
        The raw spatial data matrix with the coordinate position for each cell, indexed by sample.
    exps : dict
        The gene expression data for each coordinate in coords_raw.
    nodenum_t : int, optional (default: '20000')
        The number of nodes to return.
    if_non_zero_center_scale : bool, optional (default: 'True')
        Whether or not to scale the expression matrix by dividing each column by its standard deviation without centering (i.e. without subtracting the mean).
    
    Returns
    -------
    coords_sub : dict
        The sub-node coordinates for each sample in sample_list
    exp_sub : dict
        The sub-node expression matrix for each sample in sample_list 
    sub_node_idxs : dict
        The sub-node indices for each sample in sample_list
    """

    coords_sub = dict()
    exp_sub = dict()
    sub_node_idxs = dict()

    for sample_t in sample_list:
        exp_t,sub_node_idxs[sample_t] = sub_node_sum(coords_raw[sample_t],exps[sample_t],nodenum=nodenum_t,vis = False)
        exp_sub[sample_t] = non_zero_center_scale(exp_t.toarray()) if if_non_zero_center_scale else exp_t.toarray()
        coords_sub[sample_t] = coords_raw[sample_t][sub_node_idxs[sample_t],:]

    return coords_sub,exp_sub,sub_node_idxs

def preprocess_fast(sdata1, mode = 'customized',target_sum=1e4,base = 2,zero_center = True,regressout = False):
    """
    Convert the data to a csr matrix and preprocess it in multiple ways: total counts, log transform, scale, and regress out (if regressout).

    Parameters
    ----------
    sdata1 : AnnData
        Annotated data matrix.
    mode : 'customized' | 'default' , optional (default: 'customized')
        The mode of the preprocessing.
        If 'default', the function will use the default preprocessing settings.
        If 'customized', the user can specify the target sum, base, zero center, and regressout settings.
    target_sum : float, optional (default: '1e4')
        The target sum for the normalization (if 'mode' is 'customized').
    base : int, optional (default: '2')
        The base for the log transformation (if 'mode' is 'customized').
    zero_center : bool, optional (default: 'True')
        Whether or not to zero center the data.
    regressout : bool, optional (default: 'False')
        Whether or not to regress out the total counts.
    
    Returns
    -------
    sdata1 : AnnData
        The preprocessed data matrix
    """


    print('Preprocessing...')
    from scipy.sparse import csr_matrix as csr

    if 'raw' in sdata1.layers:
        # Data is already in raw, so convert it to a csr matrix and copy it to X 
        if type(sdata1.layers['raw']) != scipy.sparse._csr.csr_matrix:
            sdata1.layers['raw'] = csr(sdata1.layers['raw'].copy())
        sdata1.X = sdata1.layers['raw'].copy()
    else:
        # Data is not in raw, so convert it to a csr matrix and copy it to raw
        if type(sdata1.X) != scipy.sparse._csr.csr_matrix:
            sdata1.X = csr(sdata1.X.copy())
        sdata1.layers['raw'] = sdata1.X.copy()


    if mode == 'default':
        ## normalize the data in multiple ways: total counts, log transform, scale, and regress out (if regressout)
        # normalize the data to the total counts
        sc.pp.normalize_total(sdata1)
        sdata1.layers['norm'] = csr(sdata1.X.copy())

        # log transform the data
        sc.pp.log1p(sdata1)
        sdata1.layers['log1p_norm'] = csr(sdata1.X.copy())

        # scale the data to have unit variance (and possibly zero mean) as a non-csr matrix
        sc.pp.scale(sdata1,zero_center = zero_center)
        if scipy.sparse.issparse(sdata1.X): #### automatically change to non csr matrix (zero_center == True, the .X would be sparce)
            sdata1.X = sdata1.X.toarray().copy()
        sdata1.layers['log1p_norm_scaled'] = sdata1.X.copy()

        if regressout:
            # sc.pp.regress_out attempts to remove unwanted variaton from the total counts
            sdata1.obs['total_counts'] = sdata1.layers['raw'].toarray().sum(axis=1)
            sc.pp.regress_out(sdata1, ['total_counts'])
            sdata1.layers['log1p_norm_scaled'] = sdata1.X.copy()

        return sdata1 #### sdata1.X is sdata1.layers['log1p_norm_scaled']
    

    elif mode == 'customized':
        ## This customized preprocessing function allows the user to specify the target sum, base, zero center, and regressout settings

        # the data is normalized to the user provided target sum 
        if target_sum == 1e4:
            target_sum_str = '1e4'
        else:
            target_sum_str = str(target_sum)
        sc.pp.normalize_total(sdata1,target_sum=target_sum)
        sdata1.layers[f'norm{target_sum_str}'] = csr(sdata1.X.copy())

        # the data is log transformed with the user provided base
        sc.pp.log1p(sdata1,base = base)
        sdata1.layers[f'log{str(base)}_norm{target_sum_str}'] = csr(sdata1.X.copy())

        # scale the data to have unit variance (and possibly zero mean) as a non-csr matrix
        sc.pp.scale(sdata1,zero_center = zero_center)
        if scipy.sparse.issparse(sdata1.X): #### automatically change to non csr matrix (zero_center == True, the .X would be sparce)
            sdata1.X = sdata1.X.toarray().copy()
        sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled'] = sdata1.X.copy()

        if regressout:
            # sc.pp.regress_out attempts to remove unwanted variaton from the total counts
            sdata1.obs['total_counts'] = sdata1.layers['raw'].toarray().sum(axis=1)
            sc.pp.regress_out(sdata1, ['total_counts'])
            sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled'] = sdata1.X.copy()

        return sdata1 #### sdata1.X is sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled']
    
    else:
        print('Please set the `mode` as one of the {"default", "customized"}.')

def cell_select(coords_t, s=0.5, c=None, output_path_t=None):
    """
    Displays an interactive widget to select cells by drawing a polygon on the plot.
    Click the "Finish Polygon" button to finish drawing the polygon.
    Click the "Clear Polygon" button to clear the polygon.

    Parameters
    ----------
    coords_t : ndarray
        The spatial data matrix with the coordinate position for each cell.
    s : float, optional (default: '0.5')
        The size of the scatter plot points.
    c : str, optional (default: 'None')
        The color of the scatter plot points.
    output_path_t : str, optional (default: 'None')
        The path to save the output plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    import ipywidgets as widgets
    import numpy as np

    indices = np.arange(coords_t.shape[0]).reshape(-1, 1)
    coords = np.hstack((coords_t, indices))
    
    global poly_coords, polygon_patch, selected_cell_ids
    poly_coords = []
    polygon_patch = None
    selected_cell_ids = []

    def process_selected_ids(selected_ids):
        # Inner function to process the selected cell IDs
        print("Selected Cell IDs:", selected_ids)
        # Additional processing can be done here

    def on_click(event):
        # Inner function to handle the click event
        # Clicking inside the plot will add a point to the polygon 

        global poly_coords, polygon_patch
        if event.inaxes is not ax or event.button != 1: # not in the plot or not left click
            return
        poly_coords.append((event.xdata, event.ydata))

        # if the polygon patch already exists, remove it
        if polygon_patch:
            polygon_patch.remove()
        
        # draw the updated polygon patch
        polygon_patch = Polygon(poly_coords, closed=False, color='blue', alpha=0.3)
        ax.add_patch(polygon_patch)
        fig.canvas.draw()

    def finish_polygon(b):
        # Inner function to finish drawing the polygon

        global poly_coords, polygon_patch, selected_cell_ids
        if polygon_patch:
            polygon_patch.set_closed(True)
            fig.canvas.draw()

            # Get the cell IDs that are inside the polygon
            shapely_poly = ShapelyPolygon(poly_coords)
            selected_cell_ids = [int(id) for x, y, id in coords if shapely_poly.contains(Point(x, y))]
            process_selected_ids(selected_cell_ids)

            # save to output_path_t if provided
            if output_path_t is not None:
                fig.savefig(output_path_t)
            poly_coords.clear()

    def clear_polygon(b):
        # Inner function to clear the polygon
        global poly_coords, polygon_patch
        if polygon_patch:
            polygon_patch.remove()
            polygon_patch = None
            fig.canvas.draw()
            poly_coords.clear()

    fig, ax = plt.subplots(figsize=[10, 10])
    x, y, ids = zip(*coords) # Unzip the coordinates and cell IDs
    ax.scatter(x, y, s=s, c=c)
    ax.set_aspect('equal', adjustable='box')

    finish_button = widgets.Button(description="Finish Polygon")
    finish_button.on_click(finish_polygon)

    clear_button = widgets.Button(description="Clear Polygon")
    clear_button.on_click(clear_polygon)

    display(widgets.HBox([finish_button, clear_button]))

    fig.canvas.mpl_connect('button_press_event', on_click)

#### Delta analysis


def get_neighborhood_rad(coords_centroids, coords_candidate, radius_px, dist=None):
    if dist is None:
        dist =  pairwise_distances(coords_centroids, coords_candidate, metric='euclidean', n_jobs=-1)
    rad_mask = dist < radius_px
    return rad_mask

def delta_cell_cal(coords_tgt,coords_ref,ctype_tgt,ctype_ref,radius_px):
    """
    Calculate the delta cell counts between target cells and reference cells based on their coordinates and cell types.

    Parameters
    ----------
    coords_tgt : np.array
        The coordinates of niche centroids (target cells).
    coords_ref : np.array
        The coordinates of reference cells.
    ctype_tgt : np.array
        The cell types of niche centroids.
    ctype_ref : np.array
        The cell types of reference cells.
    radius_px : float
        The radius of the neighborhood.

    Returns
    -------
    df_delta_cell_tgt : pandas.DataFrame
        The raw cell type counts of target cells given niche centroids.
    df_delta_cell_ref : pandas.DataFrame
        The raw cell type counts of reference cells given niche centroids.
    df_delta_cell : pandas.DataFrame
        The delta cell counts (delta_cell_tgt - delta_cell_ref).

    Examples
    --------

    * coords_tgt = coords_final['injured']
    * coords_ref = coords_final['normal']
    * ctype_tgt = sdata.obs['Annotation'][right_idx]
    * ctype_ref = sdata.obs['Annotation'][left_idx]
    * radius_px = 1000
    * df_delta_cell_tgt, df_delta_cell_ref, df_delta_cell = delta_cell(coords_tgt, coords_ref, ctype_tgt, ctype_ref, radius_px)

    """
    ##### 1. generate nbhd_mask_tgt and nbhd_mask_ref.
    # nbhd_mask_tgt: coords_tgt vs coords_tgt itself.
    # nbhd_mask_ref: coords_tgt vs coords_ref.
    nbhd_mask_tgt = get_neighborhood_rad(coords_tgt, coords_tgt, radius_px)
    nbhd_mask_ref = get_neighborhood_rad(coords_tgt, coords_ref, radius_px)
    
    ##### 2. generate ctype_one_hot_array
    # ctype_one_hot_array: one-hot encoding of cell types.
    # To make the order of columns consistent, we stack the two labels together.
    ctype_all = np.hstack([ctype_tgt,ctype_ref])
    idx_ctype_tgt = np.arange(len(ctype_tgt))
    idx_ctype_ref = np.arange(len(ctype_tgt),len(ctype_tgt)+len(ctype_ref))
    ctype_one_hot = pd.get_dummies(ctype_all)
    ctype_one_hot_cols = ctype_one_hot.columns
    ctype_one_hot_tgt = ctype_one_hot.values[idx_ctype_tgt]
    ctype_one_hot_ref = ctype_one_hot.values[idx_ctype_ref]

    ##### 3. generate delta_cell_tgt, delta_cell_ref and delta_cell
    # delta_cell_tgt: raw cell type counts of target cells given niche centroids.
    # delta_cell_ref: raw cell type counts of reference cells given niche centroids.
    # delta_cell: delta_cell_tgt - delta_cell_ref.
    d_cell_tgt = nbhd_mask_tgt.astype(int).dot(ctype_one_hot_tgt.astype(int))
    d_cell_ref = nbhd_mask_ref.astype(int).dot(ctype_one_hot_ref.astype(int))
    d_cell = d_cell_tgt - d_cell_ref

    return pd.DataFrame(d_cell_tgt,columns = ctype_one_hot_cols), pd.DataFrame(d_cell_ref,columns = ctype_one_hot_cols), pd.DataFrame(d_cell,columns = ctype_one_hot_cols)

def delta_exp_cal(coords_tgt,coords_ref,exp_tgt,exp_ref,radius_px,valid_tgt_idx=None,valid_ref_idx=None):
    """
    Calculate the delta gene expression between target cells and reference cells based on their coordinates and gene expression.

    Parameters
    ----------
    coords_tgt : np.array
        The coordinates of niche centroids (target cells).
    coords_ref : np.array
        The coordinates of reference cells.
    exp_tgt : np.array
        The gene expression of target cells.
    exp_ref : np.array
        The gene expression of reference cells.
    radius_px : float
        The radius of the neighborhood.

    Returns
    -------
    df_delta_exp_tgt : np.array
    delta_exp_ref : np.array
    delta_exp : np.array
    """
    ##### 0. generate valid_tgt_idx and valid_ref_idx. For the cell type specific analysis, we only consider the cells in the given cell type.
    valid_tgt_idx = np.arange(len(coords_tgt)) if valid_tgt_idx is None else valid_tgt_idx
    valid_ref_idx = np.arange(len(coords_ref)) if valid_ref_idx is None else valid_ref_idx

    ##### 1. generate nbhd_mask_tgt and nbhd_mask_ref.
    # nbhd_mask_tgt: coords_tgt vs coords_tgt itself.
    # nbhd_mask_ref: coords_tgt vs coords_ref.
    nbhd_mask_tgt = get_neighborhood_rad(coords_tgt, coords_tgt[valid_tgt_idx], radius_px)
    nbhd_mask_ref = get_neighborhood_rad(coords_tgt, coords_ref[valid_ref_idx], radius_px)

    ##### 2. generate delta_cell_tgt, delta_cell_ref and delta_cell
    # delta_exp_tgt: Average gene expression of target cells given niche centroids.
    # delta_exp_ref: Average gene expression of reference cells given niche centroids.
    # delta_exp: delta_exp_tgt - delta_exp_ref.
    d_exp_tgt = nbhd_mask_tgt.dot(exp_tgt[valid_tgt_idx]).astype(float) / nbhd_mask_tgt.sum(axis=1)[:,None]
    d_exp_ref = nbhd_mask_ref.dot(exp_ref[valid_ref_idx]).astype(float) / nbhd_mask_ref.sum(axis=1)[:,None]

    ### if the nbhd_mask_tgt.sum(axis=1)[:,None] is 0, then the d_exp_tgt, d_exp_ref will be nan. We set it to 0.
    d_exp_tgt[np.isnan(d_exp_tgt)] = 0 
    d_exp_ref[np.isnan(d_exp_ref)] = 0
    d_exp = d_exp_tgt - d_exp_ref

    return d_exp_tgt, d_exp_ref, d_exp

def delta_exp_sigplot(p_values,avg_differences,abs_10logp_cutoff = None, abs_avg_diff_cutoff = None, sig = True):
    y_t = np.array(-np.log10(p_values))
    x_t = np.array(avg_differences)
    abs_10logp_cutoff = np.quantile(np.abs(y_t),0.95) if abs_10logp_cutoff is None else abs_10logp_cutoff
    abs_avg_diff_cutoff = np.quantile(np.abs(x_t),0.95) if abs_avg_diff_cutoff is None else abs_avg_diff_cutoff
    idx_sig = (np.abs(y_t) > abs_10logp_cutoff) & (np.abs(x_t) > abs_avg_diff_cutoff) if sig else np.zeros(len(y_t),dtype = bool)
    idx_sig_up = (y_t > abs_10logp_cutoff) & (x_t > abs_avg_diff_cutoff) if sig else np.zeros(len(y_t),dtype = bool)
    idx_sig_down = (y_t > abs_10logp_cutoff) & (x_t < -abs_avg_diff_cutoff) if sig else np.zeros(len(y_t),dtype = bool)
    plt.figure(figsize = (10,10))
    plt.scatter(x_t,y_t, s = 2, c = 'black', rasterized = True)
    plt.scatter(x_t[idx_sig],y_t[idx_sig], s = 5, c = 'red', rasterized = True)
    plt.xlabel('Average difference')
    plt.ylabel('-log10(p)')
    return idx_sig, idx_sig_up, idx_sig_down

def delta_exp_statistics(delta_exp_tgt, delta_exp_ref):
    from scipy.stats import ranksums
    from tqdm import tqdm
    p_values = []
    avg_differences = []
    for i in tqdm(range(delta_exp_tgt.shape[1])):
        # Calculate the rank-sum p-value
        p_value = ranksums(delta_exp_tgt[:, i], delta_exp_ref[:, i]).pvalue
        p_values.append(p_value)
        # Calculate the average of the differences
        avg_difference = np.mean(delta_exp_tgt[:, i] - delta_exp_ref[:, i])
        avg_differences.append(avg_difference)
    return p_values, avg_differences