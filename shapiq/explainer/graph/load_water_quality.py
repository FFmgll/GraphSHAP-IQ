from .graph_datasets import WaterQuality
from .graph_models import QualityModel
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import os, torch

CKPT_PATH = 'shapiq/explainer/graph/ckpt/graph_prediction/QualityModel/WaterQuality/'

def load_quality_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QualityModel(
        in_channels=1, hidden_channels=16, out_channels=1, n_layers=3
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_PATH, 'QualityModel.pt')))
    
    return model

def load_water_quality_data():
    # Training set
    ds = WaterQuality(subset='train')
    ds = DataLoader(ds, batch_size=64, shuffle=True)

    # Testing set
    ds_test = WaterQuality(subset='test')
    ds_test = DataLoader(ds_test, batch_size=64)
    return {
        'train' : ds,
        'test' : ds_test
    }



if __name__ == '__main__':
    '''
    To call the main function, use:
    > python -m shapiq.explainer.graph.load_water_quality

    A sample from WaterQuality dataset:

    x (Node Features) => shape: [32, 1] -> [#Nodes, #NodeFeatures]
    edge_index        => shape: [2, 68] -> [2, #Edges]
    edge_features     => shape: [68, 1] -> [#Edges, #EdgeFeatures]

    DataLoader is required! (To use without batching, set batch_size=1)
    '''
    model = load_quality_model()
    model.eval()

    ds_test = load_water_quality_data()['test']
    # Sample a batch from dataset
    sample = next(iter(ds_test))
    # Call the model prediction.shape: [BatchSize, 1]
    with torch.no_grad():
        predicted_chlorination = model(sample.x, sample.edge_index, sample.edge_features, sample.batch)
    # Loss computation
    test_loss = F.l1_loss(sample.label, predicted_chlorination).cpu().numpy().item()
    print(f'Model achieves {test_loss:.4f} MAE on the test set.')
