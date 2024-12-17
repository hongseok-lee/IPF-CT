import torch
import torch.nn as nn


class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob

class Cumulative_Probability_Layer_Plus(nn.Module):
    def __init__(self, num_features, args, max_followup, n_clinical_features=0):
        super(Cumulative_Probability_Layer_Plus, self).__init__()
        self.args = args
        self.hazard_fc1 = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc2 = nn.Sequential(
            nn.Linear(num_features+n_clinical_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc_raw = nn.Sequential(
            nn.Linear(num_features+n_clinical_features, max_followup),
            nn.BatchNorm1d(max_followup),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc_base = nn.Sequential(
            nn.Linear(num_features+n_clinical_features, 1),
            nn.BatchNorm1d(1),
        )
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x, clinical_features=None):
        x = self.hazard_fc1(x)
        x = torch.cat((x, clinical_features), dim=1)
        x = self.hazard_fc2(x)
        x = torch.cat((x, clinical_features), dim=1)
        x = self.hazard_fc2(x)
        x = torch.cat((x, clinical_features), dim=1)
        raw_hazard = self.hazard_fc_raw(x)
        base_hazard = self.hazard_fc_base(x)
        base_hazard = torch.clamp(base_hazard, min=-5, max=5)
        return raw_hazard, base_hazard
    

    def forward(self, x, clinical_features=None):
        hazards, base_hazard = self.hazards(x, clinical_features) # multiple FC with ReLU
        B, T = hazards.size()  # hazards is (B, T), T  is the max_followup
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob


class Cumulative_Probability_Layer_Plus_Volume(nn.Module):
    def __init__(self, num_features, args, max_followup, n_clinical_features=0, n_volume_features=0):
        super(Cumulative_Probability_Layer_Plus_Volume, self).__init__()
        self.args = args
        self.hazard_fc1 = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc2 = nn.Sequential(
            nn.Linear(num_features+n_clinical_features+n_volume_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc3 = nn.Sequential(
            nn.Linear(num_features+n_clinical_features+n_volume_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc_raw = nn.Sequential(
            nn.Linear(num_features+n_clinical_features+n_volume_features, max_followup),
            nn.BatchNorm1d(max_followup),
            nn.ReLU(inplace=True),
        )
        self.hazard_fc_base = nn.Sequential(
            nn.Linear(num_features+n_clinical_features+n_volume_features, 1),
            nn.BatchNorm1d(1),
        )
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x, clinical_features=None, volume_features=None):
        x = self.hazard_fc1(x)
        x = torch.cat((x, clinical_features, volume_features), dim=1)
        x = self.hazard_fc2(x)
        x = torch.cat((x, clinical_features, volume_features), dim=1)
        x = self.hazard_fc3(x)
        x = torch.cat((x, clinical_features, volume_features), dim=1)
        raw_hazard = self.hazard_fc_raw(x)
        base_hazard = self.hazard_fc_base(x)
        base_hazard = torch.clamp(base_hazard, min=-5, max=5)
        # TODO Other option
        # base_hazard = (torch.nn.functional.sigmoid(base_hazard) - 0.5) * 10
        return raw_hazard, base_hazard
    

    def forward(self, x, clinical_features=None, volume_features=None):
        hazards, base_hazard = self.hazards(x, clinical_features, volume_features) # multiple FC with ReLU
        B, T = hazards.size()  # hazards is (B, T), T  is the max_followup
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob
