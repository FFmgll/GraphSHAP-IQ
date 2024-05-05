from torch_geometric.datasets import TUDataset
from graphxai_local.datasets.dataset import GraphDataset

class CustomTUDataset(GraphDataset):
    '''
    GraphXAI implementation MUTAG dataset
        - Contains MUTAG with ground-truth

    Args:
        root (str): Root directory in which to store the dataset
            locally.
        generate (bool, optional): (:default: :obj:`False`)
    '''

    def __init__(self,
        name: str,
        root: str,
        use_fixed_split: bool = True,
        generate: bool = True,
        split_sizes = (0.7, 0.2, 0.1),
        seed = None
        ):

        self.graphs = TUDataset(root=root, name=name)
        # self.graphs retains all qualitative and quantitative attributes from PyG

        #self.__make_explanations()

        super().__init__(name = name, seed = seed, split_sizes = split_sizes)
