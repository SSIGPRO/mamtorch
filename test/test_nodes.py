import torch
import mamtorch as mam
from torchvision.models.feature_extraction import get_graph_node_names

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        self.linear = torch.nn.Linear(100, 100) 
        self.mamfullyconnected = mam.nn.FullyConnected(100, 100)

    def forward(self, x):
        # Pass input through all layers
        x = self.linear(x)
        x = self.mamfullyconnected(x)
        return x


model = MyModel() 
print(get_graph_node_names(model)[0])
