import torch, dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from models.aug import random_aug
from utils import coords2adjacentmat
from timeit import default_timer as timer
from collections import OrderedDict
from tqdm import trange

def train_seq(graphs, args, dump_epoch_list, out_prefix, model):
    """
    Trains a model_GCNII.CCA_SSG model (using the CCA-SSG approach) for CAST Mark.

    Parameters 
    ----------
    graphs : List[Tuple(str, dgl.Graph, torch.Tensor)]
        List of 3-member tuples, where each tuple represents one tissue sample. The tuple elements are the sample name, a DGL graph object, and a feature matrix.
    args : model_GCNII.Args
        The Args object contains training parameters
    dump_epoch_list : List[int]
        A list of epoch iterations you hope training snapshots to be dumped for debugging 
    out_prefix : str
        File name prefix for the snapshot files
    model : model_GCNII.CCA_SSG
        The untrained GNN model.
    
    Returns
    -------
    Tuple(Dict[str, torch.Tensor], List[float], model_GCNII.CCA_SSG)
        A dictionary containing the graph embeddings for each sample, a list of the loss value per epoch, and the trained model.
    """
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    loss_log = []
    time_now = timer()
    
    t = trange(args.epochs, desc='', leave=True) # Progress Bar
    for epoch in t:

        # Dump the embeddings for each sample if the epoch is in the dump_epoch_list
        with torch.no_grad():
            if epoch in dump_epoch_list:
                model.eval()
                dump_embedding = OrderedDict()
                for name, graph, feat in graphs:
                    # graph = graph.to(args.device)
                    # feat = feat.to(args.device)
                    dump_embedding[name] = model.get_embedding(graph, feat)
                torch.save(dump_embedding, f'{out_prefix}_embed_dict_epoch{epoch}.pt')
                torch.save(loss_log, f'{out_prefix}_loss_log_epoch{epoch}.pt')
                print(f"Successfully dumped epoch {epoch}")

        # Train the model
        losses = dict()
        model.train()
        optimizer.zero_grad()

        # For each sample, perform random augmentation and calculate the loss
        for name_, graph_, feat_ in graphs:

            # Random augmentation
            with torch.no_grad():
                N = graph_.number_of_nodes()
                graph1, feat1 = random_aug(graph_, feat_, args.dfr, args.der)
                graph2, feat2 = random_aug(graph_, feat_, args.dfr, args.der)

                graph1 = graph1.add_self_loop()
                graph2 = graph2.add_self_loop()

            z1, z2 = model(graph1, feat1, graph2, feat2)

            # Similiarity Matricies 
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            # Scaling by number of nodes
            c = c / N
            c1 = c1 / N
            c2 = c2 / N

            # Loss Calculation and optimization
            loss_inv = - torch.diagonal(c).sum()
            iden = torch.eye(c.size(0), device=args.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)
            loss.backward()
            optimizer.step()

        # Record the epoch loss     
        loss_log.append(loss.item())
        time_step = timer() - time_now
        time_now += time_step
        t.set_description(f'Loss: {loss.item():.3f} step time={time_step:.3f}s')
        t.refresh()
    
    model.eval()

    # store the embeddings for each sample
    with torch.no_grad():
        dump_embedding = OrderedDict() 
        for name, graph, feat in graphs:
            dump_embedding[name] = model.get_embedding(graph, feat)

    return dump_embedding, loss_log, model

# graph construction tools
def delaunay_dgl(sample_name, df, output_path,if_plot=True,strategy_t = 'convex'):
    """
    Constructs a delaunay graph from a given dataframe.

    Parameters
    ----------
    sample_name : str
        The name of the sample.
    df : array-like (castable to numpy array)
        An array containing the coordinates of the points.
    output_path : str
        The path to save the plot (if if_plot is True).
    if_plot : bool, optional (default: True)
        Whether to display and save the graph.
    strategy_t : 'convex' | 'delaunay', optional (default: 'convex')
        The strategy to construct the delaunay graph
        Convex will use Veronoi polygons clipped to the convex hull of the points and their rook spatial weights matrix (with libpysal).
        Delaunay will use the Delaunay triangulation (with sciipy).
    
    Returns
    -------
    dgl.DGLGraph
        The delaunay graph in the DGL format
    """

    coords = np.column_stack((np.array(df)[:,0],np.array(df)[:,1]))
    delaunay_graph = coords2adjacentmat(coords,output_mode = 'raw',strategy_t = strategy_t)

    # plot the graph and save to the output_path
    if if_plot:
        positions = dict(zip(delaunay_graph.nodes, coords[delaunay_graph.nodes,:]))
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        nx.draw(
            delaunay_graph,
            positions,
            ax=ax,
            node_size=1,
            node_color="#000000",
            edge_color="#5A98AF",
            alpha=0.6,
        )
        plt.axis('equal')
        plt.savefig(f'{output_path}/delaunay_{sample_name}.png')

    return dgl.from_networkx(delaunay_graph)