from sybil.datasets.validation import CSVDataset
from sybil.datasets.nlst import *
from sybil.datasets.snuh_dicom import SNUH_Survival_Dataset as SNUH_Survival_Dataset_Dicom
from sybil.datasets.snuh_h5 import SNUH_Survival_Dataset as SNUH_Survival_Dataset_H5
from sybil.datasets.snuh_npy import SNUH_Survival_Dataset as SNUH_Survival_Dataset_Npy
from sybil.datasets.sameday_dicom import Sameday_Survival_Dataset as Sameday_Survival_Dataset_Dicom
from sybil.datasets.mgh import MGH_Screening

SUPPORTED_DATASETS = {
    "validation": CSVDataset,
    "nlst": NLST_Survival_Dataset,
    "nlst_risk_factors": NLST_Risk_Factor_Task,
    "mgh": MGH_Screening,
    "snuh_dicom": SNUH_Survival_Dataset_Dicom,
    "snuh_h5": SNUH_Survival_Dataset_H5,
    "snuh_npy": SNUH_Survival_Dataset_Npy,
    "sameday": Sameday_Survival_Dataset_Dicom,
}


def get_dataset(dataset_name, split, args):
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError("Dataset {} does not exist.".format(dataset_name))
    return SUPPORTED_DATASETS[dataset_name](args, split)
