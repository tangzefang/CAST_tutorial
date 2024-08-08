import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os

def kmeans_plot_multiple(embed_dict_t,graph_list,coords,taskname_t,output_path_t,k=20,dot_size = 10,scale_bar_t = None,minibatch = True,plot_strategy = 'sep',axis_off = False):
    """
    Plot and return the KMeans clustering results for multiple samples.

    Parameters
    ----------
    embed_dict_t : dict
        A dictionary of the embeddings of the samples.
    graph_list : list[str]
        The list of the names of the samples.
    coords : dict
        A dictionary of the coordinates of the samples (str keys and array-like values).
    taskname_t : str
        The name of the task for the output file name ({taskname_t}_trained_k{str(k)}.pdf)
    output_path_t : str
        The path to save the plot.
    k : int (default = 20)
        The number of clusters for KMeans.
    dot_size : int (default = 10)
        The size of the points in the plot.
    scale_bar_t : list[float]
        The length (in data units) and label of the scale bar for the plot. If omitted, no scale bar is added.
    minibatch : bool (default = True)
        Whether to use MiniBatchKMeans for clustering.
    plot_strategy : str (default = 'sep')
        The strategy to plot the clustering results. If 'sep', each sample is plotted in a separate subplot. Else, all samples are plotted in the same subplot.
    axis_off : bool (default = False)
        Whether to turn off the axis in the plot.
    
    Returns
    -------
    cell_label : np.array
        The labels of the cells in the KMeans clustering.
    """
    num_plot = len(graph_list)
    plot_row = int(np.floor(num_plot/2) + 1)
    embed_stack = embed_dict_t[graph_list[0]].cpu().detach().numpy()
    for i in range(1,num_plot):
        embed_stack = np.row_stack((embed_stack,embed_dict_t[graph_list[i]].cpu().detach().numpy()))

    print(f'Perform KMeans clustering on {embed_stack.shape[0]} cells...')
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(n_clusters=k,random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('tab20',len(np.unique(cell_label)))

    print(f'Plotting the KMeans clustering results...')
    cell_label_idx = 0
    if plot_strategy == 'sep':
        # plot each sample in a separate subplot
        plt.figure(figsize=((20,10 * plot_row)))
        for j in range(num_plot):

            # prepare the data for plotting
            plt.subplot(plot_row,2,j+1)
            coords0 = coords[graph_list[j]]
            col=coords0[:,0].tolist()
            row=coords0[:,1].tolist()
            cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
            cell_label_idx += coords0.shape[0]

            # plot the cells in different clusters with different colors
            for i in set(cell_type_t):
                plt.scatter(np.array(col)[cell_type_t == i],
                np.array(row)[cell_type_t == i], s=dot_size,edgecolors='none',
                c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i), rasterized=True)

            # set the plot title, axis, and scale bar per subplot
            plt.title(graph_list[j] + ' (KMeans, k = ' + str(k) + ')',fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.axis('equal')
            if axis_off:
                plt.xticks([])
                plt.yticks([])
            if (type(scale_bar_t) != type(None)):
                add_scale_bar(scale_bar_t[0],scale_bar_t[1])

    else:

        # plot all samples in the same plot
        plt.figure(figsize=[10,12])
        plt.rcParams.update({'font.size' : 10,'axes.titlesize' : 20,'pdf.fonttype':42})
        for j in range(num_plot):
            coords0 = coords[graph_list[j]]
            col=coords0[:,0].tolist()
            row=coords0[:,1].tolist()
            cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
            cell_label_idx += coords0.shape[0]
            for i in set(cell_type_t):
                plt.scatter(np.array(col)[cell_type_t == i],
                np.array(row)[cell_type_t == i], s=dot_size,edgecolors='none',alpha = 0.5, # make the points semi-transparent
                c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i), rasterized=True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis('equal')
        if axis_off:
            plt.xticks([])
            plt.yticks([])
        plt.title('K means (k = ' + str(k) + ')',fontsize=30)      
        if (type(scale_bar_t) != type(None)):
            add_scale_bar(scale_bar_t[0],scale_bar_t[1])  
    plt.savefig(f'{output_path_t}/{taskname_t}_trained_k{str(k)}.pdf',dpi = 100)
    return cell_label

def add_scale_bar(length_t,label_t):
    """
    Add scale bar to the current matplotlib plot
    
    Parameters
    ----------
    length_t : float
        The length of the scale bar in data coordinates
    label_t : str
        The label of the scale bar
    """

    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    fontprops = fm.FontProperties(size=20, family='Arial')
    bar = AnchoredSizeBar(plt.gca().transData, length_t, label_t, 4, pad=0.1,
                        sep=5, borderpad=0.5, frameon=False,
                        size_vertical=0.1, color='black',fontproperties = fontprops)
    plt.gca().add_artist(bar)

def plot_mid_v2(coords_q,coords_r = None,output_path='',filename = None,title_t = ['ref','query'],s_t = 8,scale_bar_t = None):
    """
    Plot the coordinates of one or two samples in the same plot.

    Parameters
    ----------
    coords_q : array-like (will be converted to np.array)
        The coordinates of the query sample.
    coords_r : array-like, optional (will be converted to np.array)
        The coordinates of the reference sample. If omitted, only the query sample is plotted.
    output_path : str
        The path to save the plot.
    filename : str
        The name of the file to save the plot.
    title_t : list[str] (default = ['ref','query'])
        The labels for the samples in the plot.
    s_t : int (default = 8)
        The size of the points in the plot.
    scale_bar_t : list[float]
        The length (in data units) and label of the scale bar for the plot. If omitted, no scale bar is added.
    """

    plt.rcParams.update({'font.size' : 30,'axes.titlesize' : 30,'pdf.fonttype':42,'legend.markerscale' : 5})
    plt.figure(figsize=[10,12])

    # plot the reference sample if provided
    if coords_r is not None:
        plt.scatter(np.array(coords_r)[:,0].tolist(),
            np.array(coords_r)[:,1].tolist(),  s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
            c='#9295CA',label = title_t[0])
        
    # plot the query sample
    plt.scatter(np.array(coords_q)[:,0].tolist(),
        np.array(coords_q)[:,1].tolist(), s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
        c='#E66665',label = title_t[1])
    
    # add the legend, set the aspect ratio, and save the plot
    plt.legend(fontsize=15)
    plt.axis('equal')
    if (type(scale_bar_t) != type(None)):
        add_scale_bar(scale_bar_t[0],scale_bar_t[1])
    if (filename != None):
        plt.savefig(os.path.join(output_path,filename + '.pdf'),dpi = 300)

def plot_mid(coords_q,coords_r,output_path='',filename = None,title_t = ['ref','query'],s_t = 8,scale_bar_t = None,axis_off = False):
    """
    Plot the coordinates of two samples in the same plot.

    Parameters
    ----------
    coords_q : array-like (will be converted to np.array)
        The coordinates of the query sample.
    coords_r : array-like (will be converted to np.array)
        The coordinates of the reference sample.
    output_path : str
        The path to save the plot.
    filename : str
        The name of the file to save the plot.
    title_t : list[str] (default = ['ref','query'])
        The labels for the samples in the plot.
    s_t : int (default = 8)
        The size of the points in the plot.
    scale_bar_t : list[float]
        The length (in data units) and label of the scale bar for the plot. If omitted, no scale bar is added.
    axis_off : bool (default = False)
        Whether to turn off the axis in the plot.
    """

    plt.rcParams.update({'font.size' : 30,'axes.titlesize' : 30,'pdf.fonttype':42,'legend.markerscale' : 5})
    plt.figure(figsize=[10,12])

    # plot the reference sample
    plt.scatter(np.array(coords_r)[:,0].tolist(),
        np.array(coords_r)[:,1].tolist(),  s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
        c='#9295CA',label = title_t[0])
    
    # plot the query sample
    plt.scatter(np.array(coords_q)[:,0].tolist(),
        np.array(coords_q)[:,1].tolist(), s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
        c='#E66665',label = title_t[1])
    
    # add the legend and set the aspect ratio
    plt.legend(fontsize=15)
    plt.axis('equal')
    if axis_off:
        plt.xticks([])
        plt.yticks([])
    
    # add the scale bar and save the plot
    if (type(scale_bar_t) != type(None)):
        add_scale_bar(scale_bar_t[0],scale_bar_t[1])
    if (filename != None):
        plt.savefig(os.path.join(output_path,filename + '.pdf'),dpi = 300)

def link_plot(all_cosine_knn_inds_t,coords_q,coords_r,k,figsize_t = [15,20],scale_bar_t = None):
    """
    Plot the links between the query cells and their k nearest neighbors in the reference cells.

    Parameters
    ----------
    all_cosine_knn_inds_t : np.array
        The indices of the k nearest neighbors in the reference cells for each query cell.
    coords_q : np.array
        The coordinates of the query cells.
    coords_r : np.array
        The coordinates of the reference cells.
    k : int
        The number of nearest neighbors to visualize.
    figsize_t : list[float], optional (default = [15,20])
        The size of the plot.
    scale_bar_t : list[float], optional
        The length (in data units) and label of the scale bar for the plot. If omitted, no scale bar is added.
    """
    
    assign_mat = all_cosine_knn_inds_t
    plt.figure(figsize=figsize_t)
    coords_transfer_r = coords_r[np.unique(assign_mat),:] # select unique points in the reference sample
    coords_transfer_q = coords_q

    plt.scatter(x = coords_transfer_q[:,0],y = coords_transfer_q[:,1],s = 2,rasterized=True) # plot the query sample
    i = list(range(coords_transfer_q.shape[0]))
    j = i.copy()

    # plot the links between the query and reference samples
    for i_t in range(k):
        idx_transfer_r_link = assign_mat[:,i_t]
        coords_transfer_r_link = coords_r[idx_transfer_r_link,:]
        t1 = np.row_stack((coords_transfer_r_link[:,0],coords_transfer_r_link[:,1]))
        t2 = np.row_stack((coords_transfer_q[:,0],coords_transfer_q[:,1]))
        plt.plot([t1[0,i],t2[0,j]],[t1[1,i],t2[1,j]],'g',lw = 0.3,rasterized=True)

    # finalize the plot 
    plt.scatter(x = coords_transfer_r[:,0],y = coords_transfer_r[:,1],s = 4,c = 'red',rasterized=True) # plot the reference sample
    plt.axis('equal')
    if (type(scale_bar_t) != type(None)): 
        add_scale_bar(scale_bar_t[0],scale_bar_t[1]) # add the scale bar

    # get the number of used dots and the ratio of used dots
    used_dots_num = np.unique(assign_mat).shape[0]
    all_dots_num = np.sum(coords_r.shape[0])
    return [all_dots_num,used_dots_num,format(used_dots_num/all_dots_num,'.2f')]

def dsplot(coords0,coords_plaque_t,s_cell=10,s_plaque=40,col_cell='#999999',col_plaque='red',cmap_t = 'vlag',alpha = 1,vmax_t = None, title=None, scale_bar_200 = None, output_path_t = None, coords0_mask = None):
    if coords0_mask is not None:
        coords_other = coords0[~coords0_mask].copy()
        coords0 = coords0[coords0_mask].copy()
    else:
        coords_other = None
    if type(vmax_t) == type(None):
        vmax_t = np.abs(col_cell).max()
    plt.figure(figsize=(13,13))
    if title is not None:
        plt.title(title, fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    if type(col_cell) != str:
        col_cell_i = np.array(col_cell)[coords0_mask] if coords0_mask is not None else col_cell
    else:
        col_cell_i = col_cell
    if coords_other is not None:
        plt.scatter(coords_other[:,0], coords_other[:,1], s=s_cell, edgecolors='none',alpha = 0.2,
                    c='#aaaaaa',cmap = cmap_t,rasterized=True)
    col=coords0[:,0].tolist()
    row=coords0[:,1].tolist()
    plt.scatter(np.array(col), np.array(row), s=s_cell, edgecolors='none',alpha = alpha, vmax=vmax_t,vmin = -vmax_t,
                c=col_cell_i,cmap = cmap_t,rasterized=True)
    plt.colorbar(ticks=[-vmax_t,0, vmax_t])
    if type(coords_plaque_t) != type(None):
        coords1 = coords_plaque_t
        if type(s_plaque) != int:
            s_plaque_i = np.array(s_plaque)
        else:
            s_plaque_i = s_plaque
        if type(col_plaque) != str:
            col_plaque_i = np.array(col_plaque)
        else:
            col_plaque_i = col_plaque
        col=coords1[:,0].tolist()
        row=coords1[:,1].tolist()
        plt.scatter(np.array(col),np.array(row), s=s_plaque_i, edgecolors='none', c=col_plaque_i,rasterized=True)
    if scale_bar_200 is not None:
        add_scale_bar(scale_bar_200,'200 µm')
    if output_path_t is not None:
        plt.savefig(output_path_t,dpi = 300)
        plt.close('all')
