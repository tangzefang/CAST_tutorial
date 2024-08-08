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
    #: The GPU id, set to zero for single-GPU nodes
    gpu : int = 0 

    #: The number of epochs for training 
    epochs : int = 1000

    #: The learning rate 
    lr1 : float = 1e-3 
    #: The weight decay
    wd1 : float = 0.0 
    #: lambda in the loss function, refer to online methods
    lambd : float = 1e-3 

    #: number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
    n_layers : int = 9 

    #: edge dropout rate in CCA-SSG
    der : float = 0.2 
    #: feature dropout rate in CCA-SSG
    dfr : float = 0.2 

    #: perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
    device : str = field(init=False) 

    #: encoder dimension, ignore if `use_encoder` set to `False`
    encoder_dim : int = 256 
    #: whether or not to use an encoder
    use_encoder : bool = False 

    def __post_init__(self):
        if self.gpu != -1 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(self.gpu)
        else:
            self.device = 'cpu'


# fix the div zero standard deviation bug, Shuchen Luo (20220217)
def standardize(x, eps = 1e-12):
    """
    Standardizes the input features (subtracts the mean and divides by std).

    Parameters
    ----------
    x : torch.Tensor
        The input features
    eps : float
        The epsilon value to prevent division by zero
    
    Returns
    -------
    torch.Tensor
        The standardized features
    """
    
    return (x - x.mean(0)) / x.std(0).clamp(eps)

class Encoder(nn.Module):
    """
    A class representing an encoder model with a linear layer and ReLU activation function.
    """
    def __init__(self, in_dim : int, encoder_dim : int):
        """
        Parameters 
        ----------
        in_dim : int
            The number of input features
        encoder_dim : int
            The number of output features
        """
        super().__init__()
        self.layer = nn.Linear(in_dim, encoder_dim, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.layer(x))


# GCN2Conv(in_feats, layer, alpha=0.1, lambda_=1, project_initial_features=True, allow_zero_in_degree=False, bias=True, activation=None)
class GCNII(nn.Module):
    """
    A class representing a GCNII model - a model for self-supervised represenation learning with graph data.
    The model consists of an optional encoder, followed by n_layers GCN2Conv layers, where the first layer is passed to every layer.
    """

    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, alpha=None, lambda_=None, use_encoder=False):
        """
        Parameters 
        ----------
        in_dim : int
            The number of input features
        encoder_dim : int
            The number of output features of the encoder
        n_layers : int
            The number of GCNII layers
        alpha : List[float] (default: 0.1 for each layer)
            The alpha values for each layer
        lambda_ : List[float] (default: 1 for each layer)
            The lambda values for each layer
        use_encoder : bool
            Whether or not to use an encoder
        
        """

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
            The input graph
        x : torch.Tensor
            The input features
        
        Returns
        -------
        torch.Tensor
            The output features after the forward pass
        """

        if self.use_encoder:
            x = self.encoder(x)

        feat0 = x # pass the first layer features into every layer 
        for i in range(self.n_layers):
            x = self.relu(self.convs[i](graph, x, feat0))
        return x

    
class GCN(nn.Module):
    """
    A class representing a GCN model - a model for self-supervised represenation learning with graph data.
    The model consists of an optional encoder, followed by n_layers GraphConv layers
    """
    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, use_encoder=False):
        """
        
        Parameters
        ----------
        in_dim : int
            The number of input features
        encoder_dim : int
            The number of output features of the encoder
        n_layers : int
            The number of GCN layers
        use_encoder : bool
            Whether or not to use an encoder
        """
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
    backbone : GCNII | GCN
        The backbone of the model, either GCNII or GCN.
    
    Methods
    -------
    get_embedding(graph, feat)
        Returns the embeddings of the input features.
    forward(graph1, feat1, graph2, feat2)
        Returns the standardized embeddings of the input features after a foward pass through the backbone model.
    """
    def __init__(self, in_dim, encoder_dim, n_layers, backbone='GCNII', alpha=None, lambda_=None, use_encoder=False):
        super().__init__()
        if backbone == 'GCNII':
            self.backbone = GCNII(in_dim, encoder_dim, n_layers, alpha, lambda_, use_encoder)
        elif backbone == 'GCN':
            self.backbone = GCN(in_dim, encoder_dim, n_layers, use_encoder)

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
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

