from CAST_Mark import *
from CAST_Stack import *
from CAST_Projection import *
from utils import *
from visualize import *
from models.model_GCNII import Args, CCA_SSG

def CAST_MARK(coords_raw_t,exp_dict_t,output_path_t,task_name_t = None,gpu_t = None,args = None,epoch_t = None, if_plot = True, graph_strategy = 'convex'):
    """
    Run CAST Mark — CAST Mark captures common spatial features across multiple samples by training a self-supervised graph neural network model on the delaunay graphs of the samples.
    Save the embeddings, log, and model files to the output path, as well as the delaunay graphs for each sample. 
    Print the training log and delaunay graphs for each sample if if_plot is True. 
 
    Parameters
    ----------
    coords_raw_t : dict
        A dictionary with the sample names as keys and the spatial coordinates as values (gets cast as a np.array).
    exp_dict_t : dict
        A dictionary with the sample names as keys and the gene expression data as values (gets cast as a torch.Tensor).
    output_path_t : str
        The output path.
    task_name_t : str, optional 
        The task name, used to save the log file. The default is task1.
    gpu_t : int, optional (default: None)
        The GPU id. If None, the GPU id is set to 0 if a GPU is available.
    args : Args, optional
        The parameters for the model. If None, the default parameters are used.
    epoch_t : int, optional
        The number of epochs for training. If not provided, the number of epochs in args is used (default 400)
    if_plot : bool, optional (default: True)
        Whether or not to plot the results. 
    graph_strategy : str, optional (default: 'convex')
        The strategy to construct the delaunay graph. Options are 'convex' and 'knn'.
    
    Returns
    -------
    embed_dict : dict
        A dictionary with the sample names as keys and the embeddings as values.
    """

    ### settings
    gpu_t = 0 if torch.cuda.is_available() and gpu_t is None else -1 # GPU id
    device = 'cuda:0' if gpu_t == 0 else 'cpu'
    samples = list(exp_dict_t.keys())
    task_name_t = task_name_t if task_name_t is not None else 'task1' # naming for the output
    inputs = []

    ### construct delaunay graphs and input data
    print(f'Constructing delaunay graphs for {len(samples)} samples...')
    for sample_t in samples:
        graph_dgl_t = delaunay_dgl(sample_t,coords_raw_t[sample_t],output_path_t,if_plot=if_plot,strategy_t = graph_strategy).to(device)
        feat_torch_t = torch.tensor(exp_dict_t[sample_t], dtype=torch.float32, device=device)
        inputs.append((sample_t, graph_dgl_t, feat_torch_t))
    
    ### parameters setting
    if args is None:
        args = Args(
            dataname=task_name_t, # name of the dataset, used to save the log file
            gpu = gpu_t, # gpu id, set to zero for single-GPU nodes
            epochs=400, # number of epochs for training
            lr1= 1e-3, # learning rate
            wd1= 0, # weight decay
            lambd= 1e-3, # lambda in the loss function, refer to online methods
            n_layers=9, # number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
            der=0.5, # edge dropout rate in CCA-SSG
            dfr=0.3, # feature dropout rate in CCA-SSG
            use_encoder=True, # perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
            encoder_dim=512, # encoder dimension, ignore if `use_encoder` set to `False`
        )
    args.epochs = epoch_t if epoch_t is not None else args.epochs

    ### Initialize the model
    in_dim = inputs[0][-1].size(-1)
    model = CCA_SSG(in_dim=in_dim, encoder_dim=args.encoder_dim, n_layers=args.n_layers, use_encoder=args.use_encoder).to(args.device)

    ### Training
    print(f'Training on {args.device}...')
    embed_dict, loss_log, model = train_seq(graphs=inputs, args=args, dump_epoch_list=[], out_prefix=f'{output_path_t}/{task_name_t}_seq_train', model=model)

    ### Saving the results
    torch.save(embed_dict, f'{output_path_t}/demo_embed_dict.pt')
    torch.save(loss_log, f'{output_path_t}/demo_loss_log.pt')
    torch.save(model, f'{output_path_t}/demo_model_trained.pt')
    print(f'Finished.')
    print(f'The embedding, log, model files were saved to {output_path_t}')
    return embed_dict

def CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist= None,tmp1_f1_idx = None, mid_visual = False, sub_node_idxs = None, rescale = False, corr_q_r = None, if_embed_sub = False, early_stop_thres = None):
    """
    Run CAST Stack - CAST Stack aligns the spatial coordinates of the samples in the graph_list based on the graph embeddings through gradient-descent-based rigid registration and free-form deformation (FFD). 
    Saves the final coordinates, intermediate results, and the registration parameters to the output path.
    Prints the intermediate results if mid_visual is True.

    Parameters
    ----------
    coords_raw : dict
        A dictionary with the sample names as keys and the spatial coordinates as values (as np.arrays).
    embed_dict : dict
        A dictionary with the sample names as keys and the embeddings as values (as torch.Tensors).
    output_path : str
        The output folder path.
    graph_list : list[str]
        The list of the sample names [query sample, reference sample].
    params_dist : reg_params, optional
        The registration parameters. If omitted, the default parameters are used.
        See demo 2 for a full description of the parameters.
    tmp1_f1_idx : array-like, optional 
        The attention region for the FFD (The True/False index of all the cells of the query sample)
        If tmp1_f1_idx or params_dist is omitted, no attention region is used  
    mid_visual : bool, optional (default: False)
        Whether to plot the intermediate results.
    sub_node_idxs : dict, optional
        A dictionary with the sample names as keys and the values are bitmasks indicating whether each cell should be used for alignment (as np.arrays).
        If omitted, use all coordinates for the alignment.
    rescale : bool, optional (default: False)
        Whether to rescale the coordinates (by 22340 / the sample max).
    corr_q_r : np.array, optional
        The correlation matrix between the query and reference graph embeddings. If omitted, the correlation matrix is calculated.
    if_embed_sub : bool, optional (default: False)
        Whether to use a subset of the embeddings (defined by sub)nodes_idxs).
    early_stop_thres : float, optional
        The early stopping threshold for detecting a plateau in affine gradient descent. If omitted, no early stopping is done.
    
    Returns
    -------
    coords_final : dict
        A dictionary with the sample names as keys and the final coordinates as values (as np.arrays).
    """

    ### setting parameters
    query_sample = graph_list[0]
    ref_sample = graph_list[1]
    prefix_t = f'{query_sample}_align_to_{ref_sample}'
    result_log = dict()
    coords_raw, result_log['ref_rescale_factor'] = rescale_coords(coords_raw,graph_list,rescale = rescale)

    if sub_node_idxs is None: # use all coordinates
        sub_node_idxs = {
            query_sample: np.ones(coords_raw[query_sample].shape[0],dtype=bool),
            ref_sample: np.ones(coords_raw[ref_sample].shape[0],dtype=bool)
        }

    if params_dist is None:
        params_dist = reg_params(dataname = query_sample,
                                    gpu = 0,
                                    #### Affine parameters
                                    iterations=500,
                                    dist_penalty1=0,
                                    bleeding=500,
                                    d_list = [3,2,1,1/2,1/3],
                                    attention_params = [None,3,1,0],
                                    #### FFD parameters                                    
                                    dist_penalty2 = [0],
                                    alpha_basis_bs = [500],
                                    meshsize = [8],
                                    iterations_bs = [400],
                                    attention_params_bs = [[tmp1_f1_idx,3,1,0]],
                                    mesh_weight = [None])
    if params_dist.alpha_basis == []:
        params_dist.alpha_basis = torch.Tensor([1/3000,1/3000,1/100,5,5]).reshape(5,1).to(params_dist.device)
    round_t = 0
    plt.rcParams.update({'pdf.fonttype':42})
    plt.rcParams['axes.grid'] = False

    ### Generate correlation matrix of the graph embedding
    if corr_q_r is None: 
        if if_embed_sub:
            corr_q_r = corr_dist(embed_dict[query_sample].cpu()[sub_node_idxs[query_sample]], embed_dict[ref_sample].cpu()[sub_node_idxs[ref_sample]]) 
        else:
            corr_q_r = corr_dist(embed_dict[query_sample].cpu(), embed_dict[ref_sample].cpu())
    else:
        corr_q_r = corr_q_r
    
    # Plot initial coordinates
    kmeans_plot_multiple(embed_dict,graph_list,coords_raw,prefix_t,output_path,k=15,dot_size = 10) if mid_visual else None
    corr_heat(coords_raw[query_sample][sub_node_idxs[query_sample]],coords_raw[ref_sample][sub_node_idxs[ref_sample]],corr_q_r,output_path,filename=prefix_t+'_corr') if mid_visual else None
    plot_mid(coords_raw[query_sample],coords_raw[ref_sample],output_path,f'{prefix_t}_raw')

    ### Initialize the coordinates and tensor
    corr_q_r = torch.Tensor(corr_q_r).to(params_dist.device)
    params_dist.mean_q = coords_raw[query_sample].mean(0)
    params_dist.mean_r = coords_raw[ref_sample].mean(0)
    coords_query = torch.Tensor(coords_minus_mean(coords_raw[query_sample])).to(params_dist.device)
    coords_ref = torch.Tensor(coords_minus_mean(coords_raw[ref_sample])).to(params_dist.device)

    ### Pre-location
    theta_r1_t = prelocate(coords_query,coords_ref,max_minus_value_t(corr_q_r),params_dist.bleeding,output_path,d_list=params_dist.d_list,prefix = prefix_t,index_list=[sub_node_idxs[k_t] for k_t in graph_list],translation_params = params_dist.translation_params,mirror_t=params_dist.mirror_t)
    params_dist.theta_r1 = theta_r1_t
    coords_query_r1 = affine_trans_t(params_dist.theta_r1,coords_query)
    plot_mid(coords_query_r1.cpu(),coords_ref.cpu(),output_path,prefix_t + '_prelocation') if mid_visual else None ### consistent scale with ref coords

    ### Affine
    output_list = Affine_GD(coords_query_r1,
                        coords_ref,
                        max_minus_value_t(corr_q_r),
                        output_path,
                        params_dist.bleeding,
                        params_dist.dist_penalty1,
                        alpha_basis = params_dist.alpha_basis,
                        iterations = params_dist.iterations,
                        prefix=prefix_t,
                        attention_params = params_dist.attention_params,
                        coords_log = True,
                        index_list=[sub_node_idxs[k_t] for k_t in graph_list],
                        mid_visual = mid_visual,
                        early_stop_thres = early_stop_thres,
                        ifrigid=params_dist.ifrigid)

    similarity_score,it_J,it_theta,coords_log = output_list
    params_dist.theta_r2 = it_theta[-1]
    result_log['affine_J'] = similarity_score
    result_log['affine_it_theta'] = it_theta
    result_log['affine_coords_log'] = coords_log
    result_log['coords_ref'] = coords_ref

    # Affine results
    affine_reg_params([i.cpu().numpy() for i in it_theta],similarity_score,params_dist.iterations,output_path,prefix=prefix_t)# if mid_visual else None
    if if_embed_sub:
        embed_stack_t = np.row_stack((embed_dict[query_sample].cpu().detach().numpy()[sub_node_idxs[query_sample]],embed_dict[ref_sample].cpu().detach().numpy()[sub_node_idxs[ref_sample]]))
    else:
        embed_stack_t = np.row_stack((embed_dict[query_sample].cpu().detach().numpy(),embed_dict[ref_sample].cpu().detach().numpy()))
    coords_query_r2 = affine_trans_t(params_dist.theta_r2,coords_query_r1)
    register_result(coords_query_r2.cpu().detach().numpy(),
                    coords_ref.cpu().detach().numpy(),
                    max_minus_value_t(corr_q_r).cpu(),
                    params_dist.bleeding,
                    embed_stack_t,
                    output_path,
                    k=20,
                    prefix=prefix_t,
                    scale_t=1,
                    index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
    
    if params_dist.iterations_bs[round_t] != 0:
        ### B-Spline free-form deformation 
        padding_rate = params_dist.PaddingRate_bs # by default, 0
        coords_query_r2_min = coords_query_r2.min(0)[0] # The x and y min of the query coords
        coords_query_r2_tmp = coords_minus_min_t(coords_query_r2) # min of the x and y is 0
        max_xy_tmp = coords_query_r2_tmp.max(0)[0] # max_xy without padding
        adj_min_qr2 = coords_query_r2_min - max_xy_tmp * padding_rate # adjust the min_qr2
        setattr(params_dist,'img_size_bs',[(max_xy_tmp * (1+padding_rate * 2)).cpu()]) # max_xy
        params_dist.min_qr2 = [adj_min_qr2]
        t1 = BSpline_GD(coords_query_r2 - params_dist.min_qr2[round_t],
                        coords_ref - params_dist.min_qr2[round_t],
                        max_minus_value_t(corr_q_r),
                        params_dist.iterations_bs[round_t],
                        output_path,
                        params_dist.bleeding,
                        params_dist.dist_penalty2[round_t],
                        params_dist.alpha_basis_bs[round_t],
                        params_dist.diff_step,
                        params_dist.meshsize[round_t],
                        prefix_t + '_' + str(round_t),
                        params_dist.mesh_weight[round_t],
                        params_dist.attention_params_bs[round_t],
                        coords_log = True,
                        index_list=[sub_node_idxs[k_t] for k_t in graph_list],
                        mid_visual = mid_visual,
                        max_xy = params_dist.img_size_bs[round_t])

        # B-Spline FFD results
        register_result(t1[0].cpu().numpy(),(coords_ref - params_dist.min_qr2[round_t]).cpu().numpy(),max_minus_value_t(corr_q_r).cpu(),params_dist.bleeding,embed_stack_t,output_path,k=20,prefix=prefix_t+ '_' + str(round_t) +'_BSpine_' + str(params_dist.iterations_bs[round_t]),index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
        # register_result(t1[0].cpu().numpy(),(coords_ref - coords_query_r2.min(0)[0]).cpu().numpy(),max_minus_value_t(corr_q_r).cpu(),params_dist.bleeding,embed_stack_t,output_path,k=20,prefix=prefix_t+ '_' + str(round_t) +'_BSpine_' + str(params_dist.iterations_bs[round_t]),index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
        result_log['BS_coords_log1'] = t1[4]
        result_log['BS_J1'] = t1[3]
        setattr(params_dist,'mesh_trans_list',[t1[1]])

    ### Save results
    torch.save(params_dist,os.path.join(output_path,f'{prefix_t}_params.data'))
    torch.save(result_log,os.path.join(output_path,f'{prefix_t}_result_log.data'))
    coords_final = dict()
    _, coords_q_final = reg_total_t(coords_raw[query_sample],coords_raw[ref_sample],params_dist)
    coords_final[query_sample] = coords_q_final.cpu() / result_log['ref_rescale_factor'] ### rescale back to the original scale
    coords_final[ref_sample] = coords_raw[ref_sample] / result_log['ref_rescale_factor'] ### rescale back to the original scale
    plot_mid(coords_final[query_sample],coords_final[ref_sample],output_path,f'{prefix_t}_align')
    torch.save(coords_final,os.path.join(output_path,f'{prefix_t}_coords_final.data'))
    return coords_final

def CAST_PROJECT(sdata_inte, source_sample, target_sample, coords_source, coords_target, # integrated dataset, sample names and coordinates
    scaled_layer = 'log2_norm1e4_scaled', raw_layer = 'raw', batch_key = 'protocol', # layer names and column name of the samples in `obs`
    use_highly_variable_t = True, ifplot = True, n_components = 50, 
    umap_n_neighbors = 50, umap_n_pcs = 30, min_dist = 0.01, spread_t = 5,  # parameters for umap
    k2 = 1, source_sample_ctype_col = 'level_2', output_path = '', 
    umap_feature = 'X_umap', pc_feature = 'X_pca_harmony', integration_strategy = 'Harmony', # integration features and strategy
    ave_dist_fold = 3, save_result = True, ifcombat = True, alignment_shift_adjustment = 50, 
    color_dict = None, adjust_shift = False, metric_t = 'cosine', working_memory_t = 1000 
    ):
    """
    Run CAST Project — CAST Project projects the source sample to the target sample based on the integrated data.
    Save the projected dataset and distribution information to the output path. 
    
    Parameters
    ----------
    sdata_inte : anndata
        The integrated dataset.
    source_sample : str
        The source sample name.
    target_sample : str
        The target sample name.
    coords_source : array-like
        The coordinates of the source sample.
    coords_target : array-like
        The coordinates of the target sample.
    scaled_layer : str, optional (default: 'log2_norm1e4_scaled')
        The scaled layer name in `adata.layers`, which is used to be integrated.
    raw_layer : str, optional (default: 'raw')
        The raw layer name in `adata.layers`, which is used to be projected into target sample.
    batch_key : str, optional (default: 'protocol')
        The column name of the samples in `obs`.
    use_highly_variable_t : bool, optional (default: True)
        Whether to use highly variable genes.
    ifplot : bool, optional (default: True)
        Whether to plot the result.
    n_components : int, optional (default: 50)
        The `n_components` parameter in `sc.pp.pca`.
    umap_n_neighbors : int, optional (default: 50)
        The `n_neighbors` parameter in `sc.pp.neighbors`.
    umap_n_pcs : int, optional (default: 30)
        The `n_pcs` parameter in `sc.pp.neighbors`.
    min_dist : float, optional (default: 0.01)
        The `min_dist` parameter in `sc.tl.umap`.
    spread_t : int, optional (default: 5)
        The `spread` parameter in `sc.tl.umap`.
    k2 : int, optional (default: 1)
        Select k2 cells to do the projection for each cell.
    source_sample_ctype_col : str, optional (default: 'level_2')
        The column name of the cell type in `obs`.
    output_path : str, optional (default: '')
        The output path.
    umap_feature : str, optional (default: 'X_umap')
        The feature used for umap.
    pc_feature : str, optional (default: 'X_pca_harmony')
        The feature used for the projection.
    integration_strategy : 'Harmony' | None, optional (default: 'Harmony')
        Whether to run Harmony integration or use existing integrated features in pc_feature and umap_feature.
    ave_dist_fold : int, optional (default: 3)
        A multiplicative factor on the average distance for the physical distance threshold.
    save_result : bool, optional (default: True)
        Whether to save the results.
    ifcombat : bool, optional (default: True)
        Whether to use combat when using the Harmony integration.
    alignment_shift_adjustment : int, optional (default: 50)
        An additive factor on the average distance for the physical distance threshold. 
    color_dict : dict, optional
        The color dictionary for cell type annotation in visualizations.
    adjust_shift : bool, optional (default: False)
        Whether to adjust the shift of the source cells (using group_shift).
    metric_t : str, optional (default: 'cosine')
        The metric for the pairwise distance calculation.
    working_memory_t : int, optional (default: 1000)
        The sought maximum memory for the chunked pairwise distance calculations.
    
    Returns
    -------
    sdata_ref : anndata
        The integrated anndata object with the raw and normalized projected data as layers.
    list[np.ndarray, np.ndarray, np.ndarray, np.ndarray] 
       The indicies of the k-nearest neighbors, their corresponding weights, cosine distances and physical distances for each cell.
    """

    #### integration
    if integration_strategy == 'Harmony':
        sdata_inte = Harmony_integration(
            sdata_inte = sdata_inte,
            scaled_layer = scaled_layer,
            use_highly_variable_t = use_highly_variable_t,
            batch_key = batch_key,
            umap_n_neighbors = umap_n_neighbors,
            umap_n_pcs = umap_n_pcs,
            min_dist = min_dist,
            spread_t = spread_t,
            source_sample_ctype_col = source_sample_ctype_col,
            output_path = output_path,
            n_components = n_components,
            ifplot = True,
            ifcombat = ifcombat)
    elif integration_strategy is None:
        print(f'Using the pre-integrated data {pc_feature} and the UMAP {umap_feature}')

    #### Projection
    idx_source = sdata_inte.obs[batch_key] == source_sample
    idx_target = sdata_inte.obs[batch_key] == target_sample
    source_cell_pc_feature = sdata_inte[idx_source, :].obsm[pc_feature]
    target_cell_pc_feature = sdata_inte[idx_target, :].obsm[pc_feature]
    sdata_ref,output_list = space_project(
        sdata_inte = sdata_inte,
        idx_source = idx_source,
        idx_target = idx_target,
        raw_layer = raw_layer,
        source_sample = source_sample,
        target_sample = target_sample,
        coords_source = coords_source,
        coords_target = coords_target,
        output_path = output_path,
        source_sample_ctype_col = source_sample_ctype_col,
        target_cell_pc_feature = target_cell_pc_feature,
        source_cell_pc_feature = source_cell_pc_feature,
        k2 = k2,
        ifplot = ifplot,
        umap_feature = umap_feature,
        ave_dist_fold = ave_dist_fold,
        alignment_shift_adjustment = alignment_shift_adjustment,
        color_dict = color_dict,
        metric_t = metric_t,
        adjust_shift = adjust_shift,
        working_memory_t = working_memory_t
        )

    ### Save the results
    if save_result == True:
        sdata_ref.write_h5ad(f'{output_path}/sdata_ref.h5ad')
        torch.save(output_list,f'{output_path}/projection_data.pt')

    return sdata_ref,output_list
