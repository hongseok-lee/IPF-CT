import torch.nn as nn
import torchvision
from sybil.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybil.models.cumulative_probability_layer import Cumulative_Probability_Layer_Plus
from sybil.models.cumulative_probability_layer import Cumulative_Probability_Layer_Plus_Volume
from sybil.models.regressor import *
from sybil.models.pooling_layer import *

from sybil.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer

POOL_TYPE_TO_MODEL = {
    '123': MultiAttentionPool,
    '12': MultiAttentionPool_12, 
    '23': MultiAttentionPool_23, 
    '13': MultiAttentionPool_31, 
    '1': MultiAttentionPool_1, 
    '2': MultiAttentionPool_2, 
    '3': MultiAttentionPool_3,
    'MAPDNG': MultiAttentionPoolDualNoGMP, 
    'MAP_2D': MultiAttentionPool_2D, 
    'MAP_2D_GAP': MultiAttentionPool_2D_GAP,
    'MAP_12_2D': MultiAttentionPool_12_2D,
    'MAP_1_2D': MultiAttentionPool_1_2D,
    'MAP_sig': MultiAttentionPoolSig,
    '3D': Attention3DPool,
}

class SybilNet(nn.Module):
    def __init__(self, args):
        super(SybilNet, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])

        pool_type_str = ''.join(sorted(args.pool_type))
        self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.hidden_dim, args, max_followup=args.max_followup
        )

    def forward(self, x, batch=None):
        output = {}
        x = self.image_encoder(x)  # B, 3, 38, 24, 24
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])

        return pool_output


class ClinicalEncoder(nn.Module):
    def __init__(self):
        super(ClinicalEncoder, self).__init__()
        self.sex_fc = nn.Sequential( nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 5)) 
        self.age_ct_fc = nn.Linear(1, 1)
    
    def forward(self, sex, age_ct):
        encoded_sex = self.sex_fc(sex)
        encoded_age_ct = self.age_ct_fc(age_ct)
        
        return torch.cat((encoded_sex, encoded_age_ct), dim=1)

    
class SybilNetPlus(nn.Module):
    def __init__(self, args):
        super(SybilNetPlus, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
            
        pool_type_str = ''.join(sorted(args.pool_type))
        self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES
        )

    def forward(self, x, batch=None):
        output = {}
        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)
        
        # pool3D : 
        # {
        #    'hidden': output, # (B,C)
        #    'spatial_attention': spatial_attn.squeeze(1),  # (B, T, H, W)
        #    'channel_attention': channel_attn.squeeze(-1).squeeze(-1).squeeze(-1),  # (B, C)
        #    'attended': attended  # (B, C, T, H, W)
        #}

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2)

        return pool_output


class SybilNetVolume(nn.Module):
    def __init__(self, args):
        super(SybilNetVolume, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim)

        self.proportion_regressor = Regressor(
            self.hidden_dim, args, out_n=len(args.pred_volume)
        )
        
        
    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output["logit"] = self.proportion_regressor(pool_output["hidden"])

        return pool_output

        
    def forward(self, x, batch=None):
        output = {}
        x = self.image_encoder(x)  # B, 3, 38, 24, 24
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)

        return output
    
class SybilNetPlusVolume(nn.Module):
    def __init__(self, args):
        super(SybilNetPlusVolume, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )

    def forward(self, x, batch=None):
        output = {}
        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2)

        return pool_output
    
class SybilNetPlusVolumeSurvival(nn.Module):
    def __init__(self, args):
        super(SybilNetPlusVolumeSurvival, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus_Volume(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES, n_volume_features=VOLUME_FEATURES,
        )

    def forward(self, x, batch=None):
        output = {}
        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2)
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2, volume_features=pool_output['logit_volume'])

        return pool_output
    
class SybilNetPlusVolumeSurvivalMTL(nn.Module):
    def __init__(self, args):
        super(SybilNetPlusVolumeSurvivalMTL, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES
        )

    def forward(self, x, batch=None):
        output = {}
        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2)
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2)

        return pool_output
    
    
class SybilNetPlusVolumeSurvivalDetach(nn.Module):
    def __init__(self, args):
        super(SybilNetPlusVolumeSurvivalDetach, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus_Volume(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES, n_volume_features=VOLUME_FEATURES,
        )

    def forward(self, x, batch=None):
        output = {}
        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2)
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2, volume_features=pool_output['logit_volume'].detach())

        return pool_output
    
class SybilNetPlusVolumeSurvivalDetachDoubleDensity(nn.Module):
    def __init__(self, args):
        super(SybilNetPlusVolumeSurvivalDetachDoubleDensity, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()
            
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)
        self.sigmoid = nn.Sigmoid()
        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus_Volume(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES, n_volume_features=VOLUME_FEATURES + 1,
        )

    def forward(self, x, batch=None):
        output = {}

        x_1 = self.image_encoder(x['img'])  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**x['clinical'])
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2) # PRED_VOLUME="cropped_normal_density cropped_fibrosis_density"
        normal_density = self.sigmoid(pool_output['logit_volume'])[...,0]
        fibrosis_density = self.sigmoid(pool_output['logit_volume'])[...,1]
        fib_ratio = fibrosis_density / (normal_density + 1e-5)  # Added small epsilon to prevent division by zero
        fib_ratio = torch.clamp(fib_ratio, 0, 1e+5)  # Clip fib_ratio to [0, 1e+5]
         
        volume_features = torch.concat([pool_output['logit_volume'], fib_ratio.unsqueeze(-1)], dim=1) 
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2, volume_features=volume_features.detach())

        return pool_output
    
class SybilNetPlusVolumeSurvivalDetachDoubleDensityCAM(nn.Module):
    def __init__(self, args, clinical, output_target):
        super(SybilNetPlusVolumeSurvivalDetachDoubleDensityCAM, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        self.clinical = clinical
        self.output_target = output_target
        assert self.output_target in ['logit', 'logit_volume_normal' , 'logit_volume_fibrosis']
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()
            
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)
        self.sigmoid = nn.Sigmoid()
        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus_Volume(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES, n_volume_features=VOLUME_FEATURES + 1,
        )

    def forward(self, x, batch=None):
        output = {}

        x_1 = self.image_encoder(x)  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**self.clinical)
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)

        if self.output_target == 'logit':
            output = output[self.output_target][0]
        else:
            if self.output_target == 'logit_volume_normal':
                output = output['logit_volume'][0][:1]
            if self.output_target == 'logit_volume_fibrosis':
                output = output['logit_volume'][0][1:]
        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2) # PRED_VOLUME="cropped_normal_density cropped_fibrosis_density"
        normal_density = self.sigmoid(pool_output['logit_volume'])[...,0]
        fibrosis_density = self.sigmoid(pool_output['logit_volume'])[...,1]
        fib_ratio = fibrosis_density / (normal_density + 1e-5)  # Added small epsilon to prevent division by zero
        fib_ratio = torch.clamp(fib_ratio, 0, 1e+5)  # Clip fib_ratio to [0, 1e+5]
         
        volume_features = torch.concat([pool_output['logit_volume'], fib_ratio.unsqueeze(-1)], dim=1) 
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2, volume_features=volume_features.detach())

        return pool_output
    

    
class SybilNetPlusVolumeSurvivalDetachCAM(nn.Module):
    def __init__(self, args, clinical, output_target):
        super(SybilNetPlusVolumeSurvivalDetachCAM, self).__init__()

        self.hidden_dim = 512

        self.clinical = clinical
        self.output_target = output_target
        assert self.output_target in ['logit', 'logit_volume_normal' , 'logit_volume_fibrosis']
        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.clinical_encoder = ClinicalEncoder()
        CLINICAL_FEATURES = 6
        VOLUME_FEATURES = len(args.pred_volume)
            
        pool_type_str = ''.join(sorted(args.pool_type))
        if pool_type_str == '123':
            if len(args.mask_name) == 1:
                self.pool = MultiAttentionPool()
            elif len(args.mask_name) == 2:
                self.pool = MultiAttentionPoolDual()
            else:
                raise NotImplementedError
        else:
            self.pool = POOL_TYPE_TO_MODEL[pool_type_str]()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim + CLINICAL_FEATURES)

        self.proportion_regressor = RegressorPlus(
            self.hidden_dim + CLINICAL_FEATURES, args, out_n=len(args.pred_volume), n_clinical_features=CLINICAL_FEATURES
        )
        self.prob_of_failure_layer = Cumulative_Probability_Layer_Plus_Volume(
            self.hidden_dim + CLINICAL_FEATURES, args, max_followup=args.max_followup, n_clinical_features=CLINICAL_FEATURES, n_volume_features=VOLUME_FEATURES,
        )

    def forward(self, x, batch=None):
        output = {}
  
        x_1 = self.image_encoder(x)  # B, 3, 38, 24, 24
        x_2 = self.clinical_encoder(**self.clinical)
        
        pool_output = self.aggregate_and_classify(x_1, x_2)
        output["activ"] = x_1
        output.update(pool_output)
        if self.output_target == 'logit':
            output = output[self.output_target][0]
        else:
            if self.output_target == 'logit_volume_normal':
                output = output['logit_volume'][0][:1]
            if self.output_target == 'logit_volume_fibrosis':
                output = output['logit_volume'][0][1:]


        return output



    def aggregate_and_classify(self, x_1, x_2):
        pool_output = self.pool(x_1)

        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output['hidden'] = torch.cat((pool_output["hidden"], x_2), dim=1)
        pool_output["hidden"] = self.batchnorm(pool_output["hidden"])
        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["logit_volume"] = self.proportion_regressor(pool_output["hidden"], clinical_features=x_2)
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"], clinical_features=x_2, volume_features=pool_output['logit_volume'].detach())

        return pool_output
    
