from dataclasses import dataclass
import torch
import torch.nn as nn
# import torch.nn.functional as F
from dataclasses import dataclass, field
from dgl.nn import GCN2Conv, GraphConv

@dataclass
class Args:
    """
    A class representing the arguments for the GCNII model.
    """

    #: The name of the dataset, used to save the log file
    dataname : str 
    #: The GPU ID, set to zero for single-GPU nodes
    gpu : int = 0 

    #: The number of epochs for training
    epochs : int = 1000

    #: The learning rate
    lr1 : float = 1e-3 
    #: The weight decay
    wd1 : float = 0.0 
    #: The lambda in the loss function, refer to online methods
    lambd : float = 1e-3 


    n_layers : int = 9 
    """
        The number of GCNII layers. More layers mean a deeper model, larger reception field, at the cost of VRAM usage and computation time.
        By default, we chose the largest number of GCNII layers (n_layer = 9) as recommended by the original GCNII paper. Our experimental results on the simulation dataset and real samples S1-S8 confirm that increasing the number of layers improves the accuracy of CAST alignment, presumably due to the increased contrast and spatial resolution of learned graph embeddings in layer-shaped anatomical regions. These results confirmed that the performance gain from a deep GNN architecture is essential for high-resolution spatial alignment tasks.
    """

    #: The edge dropout rate in CCA-SSG
    der : float = 0.2 
    """
        The edge dropout rate in CCA-SSG.
        This hyperparameter controls the extent of graph edge dropout for graph augmentation in the CCA-SSG self-supervised learning model. der = 1 means complete dropout, while der = 0 means no dropout. For CAST, we used default der = 0.5, the same as the default in the CCA-SSG paper. Our sensitivity experiments showed that alignment performance is optimal from 0.3 to 0.7. We recommended users to use the default der value unless necessary.
    """

    dfr : float = 0.2 
    """
        The feature dropout rate in CCA-SSG. 
        This hyperparameter controls the extent of feature dropout for graph augmentation in the CCA-SSG self-supervised learning model. dfr = 1 means complete dropout while dfr = 0 means no dropout. For CAST, we used a default dfr = 0.3, following the CCA-SSG paper. Our parameter sensitivity experiments showed that alignment performance is optimal from 0.1 to 0.4. We recommend users to use the default dfr value unless necessary.
    """

    device : str = field(init=False) 
    """
        Set to the GPU_ID if GPU is available and gpu is not -1, otherwise set to cpu.
    """

    encoder_dim : int = 256 
    """
        The encoder dimension, ignored if `use_encoder` set to False
        The purpose of the MLP encoder is to reduce the time and space complexity of the model, especially for datasets with large gene panels. For our test set with a gene panel of 2,766 genes, results showed that encoder dimensions 256 and 512 yielded comparable and even slightly better alignment performance than the group without the MLP enocder module. Therefore, we recommend using 256 and 512 for parameter encoder_dim for the datasets with large gene panels (larger than 1,000 genes). We recommend using “No encoder” for datasets with limited gene panels (smaller than 1,000 genes).
    """


    #: Whether or not to use an encoder
    use_encoder : bool = False 

    def __post_init__(self):
        if self.gpu != -1 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(self.gpu)
        else:
            self.device = 'cpu'


# fix the div zero standard deviation bug, Shuchen Luo (20220217)
def standardize(x, eps = 1e-12):
    """
    Standardizes values in x (subtracts the mean and divides by standard deviation).

    Parameters
    ----------
    x : torch.Tensor
        The input features.
    eps : float, optional (default: 1e-12)
        An epsilon value to prevent division by zero.
    
    Returns
    -------
    torch.Tensor
        The standardized features.
    """
    
    return (x - x.mean(0)) / x.std(0).clamp(eps)

class Encoder(nn.Module):
    """
    A class representing an encoder model with a linear layer and ReLU activation function.

    Attributes 
    ----------
    in_dim : int
        The number of input features.
    encoder_dim : int
        The number of output features.
    """
    def __init__(self, in_dim : int, encoder_dim : int):
        super().__init__()
        self.layer = nn.Linear(in_dim, encoder_dim, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        """
        Performs a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            The input features.
        
        Returns
        -------
        torch.Tensor
            The output features after the forward pass.
        """
        return self.relu(self.layer(x))


# GCN2Conv(in_feats, layer, alpha=0.1, lambda_=1, project_initial_features=True, allow_zero_in_degree=False, bias=True, activation=None)
class GCNII(nn.Module):
    """
    A class representing a GCNII model.
    The model consists of an optional encoder, followed by GCN2Conv layers, where the first layer is passed to every layer.

    Attributes 
    ----------
    in_dim : int
        The number of input features.
    encoder_dim : int
        The number of output features of the encoder (ignored if use_encoder is false).
    n_layers : int
        The number of GCN2Conv layers.
    alpha : List[float] (default: 0.1 for each layer)
        The alpha values for each layer.
    lambda_ : List[float] (default: 1 for each layer)
        The lambda values for each layer.
    use_encoder : bool (default: False)
        Whether or not to use an encoder.
    """

    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, alpha=None, lambda_=None, use_encoder=False):

        super().__init__()

        self.n_layers = n_layers
        self.use_encoder = use_encoder

        if alpha is None:
            self.alpha = [0.1] * self.n_layers
        else:
            self.alpha = alpha

        if lambda_ is None:
            self.lambda_ = [1.] * self.n_layers
        else:
            self.lambda_ = lambda_

        if self.use_encoder:
            self.encoder = Encoder(in_dim, encoder_dim)
            self.hid_dim = encoder_dim
        else: self.hid_dim = in_dim

        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GCN2Conv(self.hid_dim, i + 1, alpha=self.alpha[i], lambda_=self.lambda_[i], activation=None)) # ReLU activation is used
    
    def forward(self, graph, x):
        """
        Forward pass through the GCNII model.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        x : torch.Tensor
            The input.
        
        Returns
        -------
        torch.Tensor
            The output of the forward pass.
        """

        if self.use_encoder:
            x = self.encoder(x)

        feat0 = x # pass the first layer features into every layer 
        for i in range(self.n_layers):
            x = self.relu(self.convs[i](graph, x, feat0))
        return x

    
class GCN(nn.Module):
    """
    A class representing a GCN model.
    The model consists of an optional encoder, followed by `n_layers` GraphConv layers

    Attributes
    ----------
    in_dim : int
        The number of input features.
    encoder_dim : int
        The number of output features of the encoder (ignored if `use_encoder` is false).
    n_layers : int
        The number of GraphConv layers.
    use_encoder : bool (default: False)
        Whether or not to use an encoder.
    """

    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, use_encoder=False):
        super().__init__()

        self.n_layers = n_layers

        self.use_encoder = use_encoder
        if self.use_encoder:
            self.encoder = Encoder(in_dim, encoder_dim)
            self.hid_dim = encoder_dim
        else: self.hid_dim = in_dim
        
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(GraphConv(self.hid_dim, self.hid_dim, activation=None)) # ReLU activation is used

    def forward(self, graph, x):
        """
        Forward pass through the GCN model.
        
        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        x : torch.Tensor
            The input.
        
        Returns
        -------
        torch.Tensor
            The output of the forward pass.
        """

        if self.use_encoder:
            x = self.encoder(x)
        # print('GCN forward: after encoder', torch.any(torch.isnan(x)))
        for i in range(self.n_layers):
            x = self.relu(self.convs[i](graph, x))
            # print('GCN layer', i + 1, 'is_nan', torch.any(torch.isnan(x)))
        return x        



class CCA_SSG(nn.Module):
    """
    A class representing a CCA_SSG model - a model for self-supervised represenation learning with graph data using GCNII or GCN as backbone.

    Attributes
    ----------
    in_dim : int
        The number of input features.
    encoder_dim : int
        The number of output features of the encoder (ignored if `use_encoder` is false).
    n_layers : int
        The number of layers in the model excluding the optional encoder. 
    backbone : GCNII | GCN
        The backbone of the model, either GCNII or GCN -- in initialization, provide 'GCNII' | 'GCN' as a string.
    alpha : List[float] (default: 0.1 for each layer)
        The alpha values for each layer of GCNII (ignored if `backbone` is GCN).
    lambda_ : List[float] (default: 1 for each layer)
        The lambda values for each layer of GCNII (ignored if `backbone` is GCN).
    use_encoder : bool (default: False)
        Whether or not to use an encoder.
    """

    def __init__(self, in_dim, encoder_dim, n_layers, backbone='GCNII', alpha=None, lambda_=None, use_encoder=False):
        super().__init__()
        if backbone == 'GCNII':
            self.backbone = GCNII(in_dim, encoder_dim, n_layers, alpha, lambda_, use_encoder)
        elif backbone == 'GCN':
            self.backbone = GCN(in_dim, encoder_dim, n_layers, use_encoder)

    def get_embedding(self, graph, feat):
        """
        Returns the result of a forward pass on `feat`.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input features.
        
        Returns
        -------
        torch.Tensor
            The result of the forward pass on the input features.
        """
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        """
        Returns the standardized embeddings of the input features after a forward pass through the backbone model.
    
        Parameters
        ----------
        graph1 : DGLGraph
            The first input graph.
        feat1 : torch.Tensor
            The input features for the first input.
        graph2 : DGLGraph
            The second input graph.
        feat2 : torch.Tensor
            The features for the second input.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The standardized outputs from running each input through a forward pass of the model.
        """

        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)
        # print('CCASSG forward: h1 is', torch.any(torch.isnan(h1)))
        # print('CCASSG forward: h2 is', torch.any(torch.isnan(h2)))
        z1 = standardize(h1)
        z2 = standardize(h2)
        # print('h1.std', h1.std(0))
        # print('h1-h1.mean(0)', h1 - h1.mean(0))
        # print('CCASSG forward: z1 is', torch.any(torch.isnan(z1)))
        # print('CCASSG forward: z2 is', torch.any(torch.isnan(z2)))

        return z1, z2

