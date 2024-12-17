import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, num_features, args, out_n):
        super(Regressor, self).__init__()
        self.args = args
        self.hidden_1 = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
        )
        self.final = nn.Linear(num_features, out_n)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        hidden = self.hidden_1(x)
        hidden = self.relu(hidden)
        hidden = self.hidden_2(hidden)
        hidden = self.relu(hidden)
        return self.final(hidden)
    
class RegressorPlus(nn.Module):
    def __init__(self, num_features, args, out_n, n_clinical_features=0):
        super(RegressorPlus, self).__init__()
        self.args = args
        self.hidden_1 =  nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
        )
        self.hidden_2 = nn.Sequential( 
            nn.Linear(num_features+n_clinical_features, num_features),
            nn.BatchNorm1d(num_features),
        )
        self.final = nn.Linear(num_features+n_clinical_features, out_n)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, clinical_features=None):
        hidden = self.hidden_1(x)
        hidden = self.relu(hidden)
        hidden = torch.cat((hidden, clinical_features), dim=1)
        hidden = self.hidden_2(hidden)
        hidden = self.relu(hidden)
        hidden = torch.cat((hidden, clinical_features), dim=1)
        return self.final(hidden)