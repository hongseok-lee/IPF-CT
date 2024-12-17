import os
from posixpath import split
import traceback, warnings
import pickle, json
import numpy as np
import pydicom
import torchio as tio
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils import data
from sybil.serie import Serie
from sybil.utils.loading import get_sample_loader
import pandas as pd
from sybil.datasets.utils import (
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
    VOXEL_SPACING,
)
import copy

from glob import glob
import numpy as np
import os
import pydicom as dicom
from typing import List, Dict, Union
from collections import namedtuple


LoaderParams = namedtuple(
    'LoaderParams', ['maximal_dcm_size', 'group_separator', 'preffered_orientation',
                     'viable_length_mm'])
_loader_params = LoaderParams(
    maximal_dcm_size = 1024 ** 2,
    viable_length_mm = [180, 720],
    group_separator = '0x0020000E', # series instsance UID  # https://dicom.innolitics.com/ciods/cr-image/general-series/0020000e
    preffered_orientation = [1, 0, 0, 0, 1, 0]) # (1,0,0) (0, 1, 0) 

class DICOM_Loader:
    
    def __init__(self, loader_params:namedtuple=_loader_params):
        self.params = loader_params
    
    def _get_grouped(self, path_to_dir:List) -> Dict:
        dicom_list, dicom_groups, dicom_paths, dicom_slices, = [], {}, {}, {}
        for root, dirs, files in os.walk(path_to_dir):
            if not files: continue
            dicom_list += \
                [os.path.join(root, x) for x in files 
                 if os.path.getsize(os.path.join(root, x)) \
                 < self.params.maximal_dcm_size]
        valid_dicom_list = []
        for d_path in dicom_list:
            try:
                pydicom.dcmread(d_path, force=True).ImagePositionPatient
                valid_dicom_list.append(d_path)
            except:
                pass
        dicom_list = valid_dicom_list
        dicom_list = sorted(
            dicom_list, key=lambda x: pydicom.dcmread(x, force=True).ImagePositionPatient[2]
        )
        verbose_flag = False
        for i, path_to_dcm in enumerate(dicom_list):
            d = dicom.read_file(path_to_dcm, force=True)
            d_prev =  dicom.read_file(dicom_list[i-1], force=True)
            if d.ImagePositionPatient == d_prev.ImagePositionPatient and np.array_equal(d.pixel_array, d_prev.pixel_array):
                # If there is duplicated slice
                continue
            
            if not '0x00080016' in d: continue 
                # super meta separator

            
                # ImageOrientationPatient 
                        
            if not np.array_equal(
                np.array(d.ImageOrientationPatient, dtype=np.int8), 
                self.params.preffered_orientation) and not verbose_flag:
                    verbose_flag = True
            if '1.2.840.10008.5.1.4.1.1.2' in d.SOPClassUID\
                and np.sum(
                        np.abs(np.array(d.ImageOrientationPatient, dtype=np.int8))\
                        - self.params.preffered_orientation) == 0:
                _key = d[self.params.group_separator].value
                if _key in dicom_paths:
                    dicom_paths[_key] += [path_to_dcm]
                else:
                    dicom_paths[_key] = [path_to_dcm]
                    
                if _key in dicom_slices:
                    dicom_slices[_key] += [d]
                else:
                    dicom_slices[_key] = [d]
            
        for guid, each_paths in dicom_paths.items():
            if len(each_paths) < 24: continue 
            paired = zip(each_paths, dicom_slices[guid])
            paired = sorted(
                paired, key=lambda x: float(x[1].ImagePositionPatient[2]))
            paths_, slices_ = [list(x) for x in zip(*paired)]
            dicom_groups[guid] = {
                'dicom_slices': slices_,
                'dicom_paths': paths_,
                'convolutional_kernel': 'N/A',
                'pixel_spacing': 'N/A',
                'spacing_between_slices': 'N/A',
                'slice_thickness': 'N/A',
                'dicom_pixels': 'N/A',
                'err_code': 0,}

        return dicom_groups

    def _check_validity(self, dicom_groups:Dict) -> None:
        # Use four basic rules to find invalid groups and give them err_code
        for guid, contents in dicom_groups.items():
            err_code = 0
            spacing_between_slices, counts = np.unique(np.round(np.abs(
                [contents['dicom_slices'][idx].ImagePositionPatient[2]\
                 - contents['dicom_slices'][idx+1].ImagePositionPatient[2]\
                 for idx in [4, 8, -8, -4]]), decimals=1), return_counts=True) # 균일한 거리인지 검사. 4,8 idx why? -> idx 0123는 잡영상 가능성 있음. 
            spacing_diff = np.array(
                [contents['dicom_slices'][idx].ImagePositionPatient[2]\
                 - contents['dicom_slices'][idx+1].ImagePositionPatient[2]\
                 for idx in [4, 8, -8, -4]]) 
             
            if (len(spacing_between_slices) > 1 and (np.max(spacing_diff) - np.min(spacing_diff) > 0.5) or max(spacing_between_slices) <= 1e-8):
                err_code += 1
                print(spacing_diff)
                print(spacing_between_slices)

            try:
                slice_thickness = np.unique(np.abs(
                    [contents['dicom_slices'][idx].SliceThickness\
                     for idx in [4, 8, -8, -4]]))
                if len(slice_thickness) > 1: err_code += 2

            except Exception as e:
                # replace the missing slice_thickness with s.b.s
                slice_thickness = spacing_between_slices
                err_code += 4
            
            try:
                convolutional_kernel = np.unique(
                    [str(contents['dicom_slices'][idx].ConvolutionKernel)\
                     for idx in [4, 8, -8, -4]])
                if len(convolutional_kernel) > 1: err_code += 8

            except:
                err_code += 8

            physical_length = spacing_between_slices[0]\
                * len(contents['dicom_slices'])
            if not (physical_length > self.params.viable_length_mm[0])\
                * (physical_length <= self.params.viable_length_mm[1]):
                err_code += 16
            
            contents['err_code'] = err_code
            contents['convolutional_kernel'] = convolutional_kernel[0]
            contents['spacing_between_slices'] = spacing_between_slices[0]
            contents['slice_thickness'] = slice_thickness[0]
    @staticmethod
    def get_pixels_hu(dicom_group:Dict) -> Dict:
        if dicom_group['err_code'] > 0: return dicom_group
        try:
            original_spacing = np.array(
                [dicom_group['spacing_between_slices']\
                , *dicom_group['dicom_slices'][0].PixelSpacing],
                dtype=np.float32)
            dicom_group['pixel_spacing'] = original_spacing
            img_arr = np.zeros(
                [len(dicom_group['dicom_slices']),
                 dicom_group['dicom_slices'][0].Rows, dicom_group['dicom_slices'][0].Columns],
                dtype=np.int16)
            for idx, slice_ in enumerate(dicom_group['dicom_slices']):
                intercept, slope = slice_.RescaleIntercept, slice_.RescaleSlope
                img_arr[idx] = np.int16(slope * slice_.pixel_array.astype(np.float32))
                img_arr[idx] += np.int16(intercept)
                if slice_.ImageOrientationPatient[0] == -1:
                    img_arr[idx] = img_arr[idx][:,::-1]
                if slice_.ImageOrientationPatient[4] == -1:
                    img_arr[idx] = img_arr[idx][::-1,:]
                    
            img_arr = np.array(img_arr, dtype=np.int16)
            img_arr[img_arr < -1024] = -1024
            dicom_group['dicom_pixels'] = img_arr
        
        except Exception as e:
            dicom_group['err_code'] += 32

        return dicom_group

    def load_dicom(self, path_to_dir):
        grouped = self._get_grouped(path_to_dir)
        self._check_validity(grouped)
        return {guid:self.get_pixels_hu(contents)\
                for guid, contents in grouped.items()}

""" 해도 리사이즈 하고 하는게 맞음.
def normalize_dicom(x_input, lowerbound=-1000., upperbound=600., eps=1.0e-8):
    x_input = x_input.astype(np.float16)
    x_input[x_input < lowerbound] = lowerbound
    x_input[x_input > upperbound] = upperbound
    x_input -= lowerbound
    x_input = x_input / (upperbound - lowerbound + eps)
    return np.array(x_input, dtype=np.float16)
""";
dicom_loader = DICOM_Loader()
# from sybil.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer

# CT_ITEM_KEYS = [
#     "pid",
#     "exam",
#     "series",
#     "y_seq",
#     "y_mask",
#     "time_at_event",
#     "cancer_laterality",
#     "has_annotation",
#     "origin_dataset",
# ]


class Sameday_Survival_Dataset(data.Dataset):
    def __init__(self, args, split_group):
        """
        SNUH Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(Sameday_Survival_Dataset, self).__init__()

        self.args = args
        self.split_group = 'test' # MUST test
        self._num_images = args.num_images  # number of slices in each volume # 336
        self.input_loader = get_sample_loader(split_group, args)
        self.always_resample_pixel_spacing = split_group in ["dev", "test"]

        self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
                               #[256, 256] + [200]
        )

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return
        

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        # self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        # self.corrupted_series = self.CORRUPTED_PATHS["series"]
        # self.risk_factor_vectorizer = NLSTRiskFactorVectorizer(self.args)

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed) # default: 0 seed used to generate the partition
            # self.assign_splits(self.metadata_json)

        DIR_PATHS = ['Measurement variability_이홍석선생님/JBUH1/dcm/2022/04/ST_18534814_00000001/SE_00005_00000001',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/04/ST_18595688_00000022/SE_00005_00000022',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/04/ST_18720511_00000026/SE_00005_00000026',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18728790_00000027/SE_00005_00000027',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18739600_00000028/SE_00005_00000028',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18753498_00000029/SE_00005_00000029',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18755388_00000030/SE_00005_00000030',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18794013_00000031/SE_00005_00000031',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18804850_00000032/SE_00005_00000032',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18810070_00000002/SE_00005_00000002',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18855491_00000003/SE_00005_00000003',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18887213_00000004/SE_00005_00000004',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18889299_00000005/SE_00005_00000005',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/05/ST_18925442_00000006/SE_00005_00000006',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_18999972_00000007/SE_00005_00000007',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19001965_00000008/SE_00005_00000008',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19022782_00000009/SE_00005_00000009',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19045559_00000010/SE_00005_00000010',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19046346_00000011/SE_00005_00000011',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19108638_00000012/SE_00005_00000012',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/06/ST_19188826_00000013/SE_00005_00000013',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19259239_00000014/SE_00005_00000014',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19267653_00000015/SE_00005_00000015',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19373220_00000016/SE_00005_00000016',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19373347_00000017/SE_00005_00000017',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19374878_00000018/SE_00005_00000018',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19419465_00000019/SE_00005_00000019',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/07/ST_19429698_00000020/SE_00005_00000020',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/08/ST_19450890_00000021/SE_00005_00000021',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/08/ST_19583010_00000023/SE_00005_00000023',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/08/ST_19642275_00000024/SE_00005_00000024',
            'Measurement variability_이홍석선생님/JBUH1/dcm/2022/08/ST_19689611_00000025/SE_00005_00000025',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/04/ST_18534814_00000001/SE_00010_00000001',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/04/ST_18595688_00000022/SE_00010_00000025',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/04/ST_18720511_00000026/SE_00010_00000032',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18728790_00000027/SE_00010_00000035',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18739600_00000028/SE_00010_00000036',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18753498_00000029/SE_00010_00000038',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18755388_00000030/SE_00010_00000039',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18794013_00000031/SE_00010_00000041',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18804850_00000032/SE_00010_00000043',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18810070_00000002/SE_00010_00000002',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18855491_00000003/SE_00010_00000003',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18887213_00000033/SE_00010_00000088',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18889299_00000005/SE_00010_00000005',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/05/ST_18925442_00000006/SE_00010_00000006',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_18999972_00000007/SE_00010_00000007',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19001965_00000008/SE_00010_00000008',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19022782_00000009/SE_00010_00000009',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19045559_00000010/SE_00010_00000010',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19046346_00000011/SE_00010_00000011',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19108638_00000012/SE_00010_00000012',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/06/ST_19188826_00000013/SE_00010_00000013',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19259239_00000014/SE_00010_00000014',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19267653_00000015/SE_00010_00000015',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19373220_00000016/SE_00010_00000016',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19373347_00000017/SE_00010_00000017',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19374878_00000018/SE_00010_00000019',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19419465_00000019/SE_00010_00000020',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/07/ST_19429698_00000020/SE_00010_00000022',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/08/ST_19450890_00000021/SE_00010_00000023',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/08/ST_19583010_00000023/SE_00010_00000026',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/08/ST_19642275_00000024/SE_00010_00000028',
            'Measurement variability_이홍석선생님/JBUH2/dcm/2022/08/ST_19689611_00000025/SE_00010_00000030',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/03/ST_1225_00000017/SE_00203_00000187',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/04/ST_2828_00000038/SE_00203_00000208',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/04/ST_2922_00000043/SE_00203_00000213',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/04/ST_2933_00000044/SE_00203_00000214',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/04/ST_3552_00000045/SE_00203_00000215',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/04/ST_3669_00000046/SE_00203_00000216',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/05/ST_4844_00000047/SE_00203_00000217',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/05/ST_4853_00000048/SE_00303_00000218',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/05/ST_5514_00000049/SE_00203_00000219',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/05/ST_5524_00000018/SE_00203_00000188',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/05/ST_5862_00000019/SE_00203_00000189',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/06/ST_7683_00000020/SE_00303_00000190',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_8950_00000021/SE_00203_00000191',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_9001_00000022/SE_00203_00000192',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_9011_00000023/SE_00203_00000193',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_9310_00000024/SE_00203_00000194',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_9421_00000025/SE_00303_00000195',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/07/ST_9733_00000026/SE_00203_00000196',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_1667_00000029/SE_00203_00000199',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_2129_00000030/SE_00203_00000200',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_2394_00000031/SE_00203_00000201',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_2512_00000032/SE_00203_00000202',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_2513_00000033/SE_00203_00000203',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_2760_00000034/SE_00203_00000204',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_917_00000027/SE_00203_00000197',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/08/ST_926_00000028/SE_00203_00000198',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/09/ST_4101_00000035/SE_00203_00000205',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/09/ST_4650_00000036/SE_00203_00000206',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/10/ST_4835_00000037/SE_00203_00000207',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/10/ST_4935_00000039/SE_00203_00000209',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/10/ST_5097_00000040/SE_00203_00000210',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/10/ST_5497_00000041/SE_00203_00000211',
            'Measurement variability_이홍석선생님/SNUH1/dcm/2022/10/ST_5506_00000042/SE_00203_00000212',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/03/ST_1225_00000001/SE_00403_00000001',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/04/ST_2828_00000022/SE_00403_00000022',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/04/ST_2922_00000027/SE_00403_00000027',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/04/ST_2933_00000028/SE_00403_00000028',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/04/ST_3552_00000029/SE_00403_00000029',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/04/ST_3669_00000030/SE_00403_00000030',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/05/ST_4844_00000031/SE_00403_00000031',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/05/ST_4853_00000032/SE_00503_00000032',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/05/ST_5514_00000033/SE_00403_00000033',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/05/ST_5524_00000002/SE_00403_00000002',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/05/ST_5862_00000003/SE_00403_00000003',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/06/ST_7683_00000004/SE_00503_00000004',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_8950_00000005/SE_00403_00000005',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_9001_00000006/SE_00403_00000006',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_9011_00000007/SE_00403_00000007',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_9310_00000008/SE_00403_00000008',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_9421_00000009/SE_00503_00000009',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/07/ST_9733_00000010/SE_00403_00000010',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_1667_00000013/SE_00403_00000013',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_2129_00000014/SE_00403_00000014',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_2394_00000015/SE_00403_00000015',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_2512_00000016/SE_00403_00000016',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_2513_00000017/SE_00403_00000017',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_2760_00000018/SE_00403_00000018',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_917_00000011/SE_00403_00000011',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/08/ST_926_00000012/SE_00403_00000012',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/09/ST_4101_00000019/SE_00403_00000019',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/09/ST_4650_00000020/SE_00403_00000020',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/10/ST_4835_00000021/SE_00403_00000021',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/10/ST_4935_00000023/SE_00403_00000023',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/10/ST_5097_00000024/SE_00403_00000024',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/10/ST_5497_00000025/SE_00403_00000025',
            'Measurement variability_이홍석선생님/SNUH2/dcm/2022/10/ST_5506_00000026/SE_00403_00000026']



        dataset = []

        for dir_path in tqdm(DIR_PATHS[:2]):
            sample = self.get_volume_dict(
                dir_path
            )
            dataset.append(sample)

        return dataset
    
    def get_volume_dict(
        self, dir_path
    ):
        PREPATH = '/data1/IPF_CT/data/added_20240405/'
        dir_id = '/'.join(dir_path.split('/')[-6:])
        sample = {
            "dir_path": PREPATH +  dir_path,
            "pid": dir_id,
        }


        return sample

    
    


    def get_images(self, sample):
        """
        returns a stack of transformed images by their absolute paths.
        if cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        dicom_dataset = dicom_loader.load_dicom(sample['dir_path'])
        if len(list(dicom_dataset.keys())) == 0:
            return -1

        
        windowing=[1500, -600]
        W, L = windowing
        for series_uid, contents  in dicom_dataset.items():
            images = contents['dicom_pixels'] # 512, 512, 336
            lowerb = L - W//2
            upperb = L + W//2
            images[images < lowerb] = lowerb
            images[images > upperb] = upperb
            images = images - lowerb
            images = np.uint8(images * (255 / W))
            sample['pixel_spacing'] = contents['pixel_spacing'].tolist()  # ndarray ->  list

        input_arr = self.reshape_images(images) 

        # resample pixel spacing
        resample_now = self.args.resample_pixel_spacing_prob > np.random.uniform() # in default, the 'resample_pixel_spacing_prob' is 1
        if self.always_resample_pixel_spacing or resample_now: # Positive
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            input_arr = self.padding_transform(input_arr.data)


        # convert from (C, T, H, W) to C, W, T, H
        input_arr = input_arr.repeat(3, 1, 1, 1) # TODO no need 3 ch. 
        out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)/255

        return out_dict

    def reshape_images(self, images):
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, axis=0) # 1, H, W, T = 1, 512, 512, 336
        
        
        # convert from (C, H, W, T) to (C, T, H, W)
        return images


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        try:
            item = {}
            out_dict = self.get_images(sample)
            x = out_dict["input"]

            item["x"] = x

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["pid"], traceback.print_exc()))
