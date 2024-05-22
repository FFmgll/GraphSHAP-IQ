from torch_geometric.datasets import TUDataset
from shapiq.explainer.graph.graph_datasets.datasets import GraphDataset
from graphxai_local.datasets.real_world.mutagenicity import Mutagenicity
from graphxai_local.datasets import FluorideCarbonyl, Benzene, AlkaneCarbonyl


def CustomTUDataset(name: str, root: str, seed: int = 42, split_sizes=(0.8, 0.1, 0.1)):
    """Helper function to switch between datasets (TU and GraphXAI)"""
    if name == "Mutagenicity_XAI":
        print("Loading Mutagenicity with explanations, it may take a while...")
        return Mutagenicity(
            root=root, seed=seed, split_sizes=split_sizes, test_debug=True
        )
    elif name in ["FluorideCarbonyl", "Benzene", "AlkaneCarbonyl"]:
        return eval(name)(seed=seed, split_sizes=split_sizes)
    else:
        return _CustomTUDataset(
            name=name, root=root, seed=seed, split_sizes=split_sizes
        )


class _CustomTUDataset(GraphDataset):
    """
    GraphXAI implementation MUTAG dataset
            - Contains MUTAG with ground-truth

    Args:
            root (str): Root directory in which to store the dataset
                    locally.
            generate (bool, optional): (:default: :obj:`False`)
    """

    def __init__(
        self,
        name: str,
        root: str,
        use_fixed_split: bool = True,
        generate: bool = True,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        device=None,
    ):
        self.graphs = TUDataset(root=root, name=name)
        # self.graphs retains all qualitative and quantitative attributes from PyG

        # self.__make_explanations()

        super().__init__(name=name, seed=seed, split_sizes=split_sizes)


if __name__ == "__main__":
    dataset = CustomTUDataset(name="AlkaneCarbonyl", root="./")
    # dataset = CustomTUDataset(name='AIDS', root='./')
    print(dataset)
    print(dataset[0])
    print(dataset[0].x)
    print(dataset[0].edge_index)
    print(dataset[0].y)
    print(dataset[0].y.unique(return_counts=True))
    print(dataset[0].num_nodes)
    print(dataset[0].num_edges)
    print(dataset[0].num_classes)
    print(dataset[0].num_node_features)
    print(dataset[0].num_edge_features)
    print(dataset[0].num_features)
    print(dataset[0].num_classes)
