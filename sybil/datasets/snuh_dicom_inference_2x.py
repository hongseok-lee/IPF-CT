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
from datetime import datetime
from datetime import timedelta
import h5py


import nibabel as nib
import numpy as np
import torchio as tio
from torchio.data.subject import Subject
from torchio.transforms.preprocessing.spatial.bounds_transform import BoundsTransform
import pydicom
import numpy as np
import cupy as cp

from cupyx.scipy.ndimage import zoom as cupy_zoom


def resize(dicom_pixels, original_spacing, new_spacing,interpolation='nearest'):
    resize_factor = [a / b for a, b in zip(original_spacing, new_spacing)]
    new_shape = [int(round((a * b) + 1)) for a, b \
                      in zip([s - 1 for s in dicom_pixels.shape], resize_factor)]
    zoom_factors = [a / b for a, b in zip(new_shape, dicom_pixels.shape)]
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    _resampled_dicom = cp.asarray(dicom_pixels)\
        .astype(dtype=cp.float32)
    _resampled_dicom = cupy_zoom(
        input=_resampled_dicom,
        zoom=zoom_factors,
        order=3, mode=interpolation,
        prefilter=True,
        grid_mode=False)
    resampled_dicom = cp.asnumpy(_resampled_dicom)
    del _resampled_dicom
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return resampled_dicom, zoom_factors


LoaderParams = namedtuple(
    'LoaderParams', ['maximal_dcm_size', 'group_separator', 'preffered_orientation',
                     'viable_length_mm'])
_loader_params = LoaderParams(
    maximal_dcm_size = 2*(1024 ** 2),
    viable_length_mm = [180, 720],
    group_separator = '0x0020000E', # series instsance UID  # https://dicom.innolitics.com/ciods/cr-image/general-series/0020000e
    preffered_orientation = [1, 0, 0, 0, 1, 0]) # (1,0,0) (0, 1, 0) 

class DICOM_Loader:
    
    def __init__(self, loader_params:namedtuple=_loader_params):
        self.params = loader_params
    
    def _get_grouped(self, path_to_dir:List) -> Dict:
        IOP_flag = False
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
                    # print('ImageOrientationPatient', d.ImageOrientationPatient)
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
            else:
                IOP_flag = (np.array(d.ImageOrientationPatient))
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

        return dicom_groups, IOP_flag

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
        grouped, IOP_flag = self._get_grouped(path_to_dir)
        self._check_validity(grouped)
        return {guid:{'IOP_flag': IOP_flag, **self.get_pixels_hu(contents)} \
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




class RandomCrop(BoundsTransform):
    def __init__(self, target_shape, sigma=1.0, **kwargs):
        super().__init__(target_shape, **kwargs)
        self.target_shape = target_shape
        self.sigma = sigma
        self.args_names = ['target_shape', 'sigma']

    def apply_transform(self, sample: Subject) -> Subject:
        input_shape = np.array(sample.spatial_shape)
        crop_shape = np.array(self.target_shape)

        # Ensure the crop shape is smaller than the input shape
        if np.any(crop_shape > input_shape):
            raise ValueError('Crop shape must be smaller than the input shape')

        # Generate centroid using Gaussian distribution centered on the middle of the input shape
        center = input_shape // 2
        centroid = np.array([np.random.normal(center[i], self.sigma * center[i]) for i in range(3)])
        centroid = np.clip(centroid, crop_shape//2, input_shape - crop_shape//2)

        # Calculate the starting indices for the crop
        starts = np.round(centroid).astype(int) - crop_shape//2
        ends = starts + crop_shape

        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, starts)
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin

            i0, j0, k0 = starts
            i1, j1, k1 = ends
            cropped_data = image.data[:, i0:i1, j0:j1, k0:k1].clone()
            image.set_data(cropped_data)
            image.affine = new_affine

        return sample

    def inverse(self):
        raise NotImplementedError('Inverse not implemented for RandomCrop')


class Inference_Dataset(data.Dataset):
    def __init__(self, args, df):
        """
        SNUH Dataset
        params: args - config.

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(Inference_Dataset, self).__init__()

        self.args = args
        self._num_images = args.num_images  # number of slices in each volume # 336
        self._max_followup = args.max_followup # in Sybil paper, max_followup = 6 yrs

        self.label_df = df
            # self.label_df = self.label_df[self.label_df.thickness_list <= args.slice_thickness_filter]
        self.padding_transform = tio.transforms.CropOrPad(
            # target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
            target_shape=tuple([args.num_images] +  [512,512] ), padding_mode=0
                               #[200] + [256, 256] 
        )
        self.center_crop_transform = tio.transforms.CropOrPad(
            target_shape=tuple([args.num_images] + args.img_size), padding_mode=0
                               #[200] + [256, 256]
        )
        self.center_crop_transform2 = tio.transforms.CropOrPad(
            target_shape=tuple([args.num_images] + [x*2 for x in args.img_size]), padding_mode=0
                               #[200] + [256, 256]
        )
        self.random_crop_transform = RandomCrop( target_shape=tuple([args.num_images] + args.img_size), sigma=args.random_crop_sigma)

        self.dataset = self.create_dataset()
        if len(self.dataset) == 0:
            return
        
        self.resample_transform = tio.transforms.Resample(target=(1.0, 1.0, 1.0))
        
    def create_dataset(self):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed) # default: 0 seed used to generate the partition
            # self.assign_splits(self.metadata_json)

        dataset = []
        for idx, row in tqdm(self.label_df.iterrows(), total=len(self.label_df)):
            sample = self.get_volume_dict(
                row
            )
            dataset.append(sample)

        return dataset
    
    def get_volume_dict(
        self, row
    ):
        sample = {
            "dir_path": row.dir_path,
            "puid": row.ID,
        }
        if 'plus' in self.args.model_type:
            if row['sex'] == 'male':
                sample['sex'] = torch.Tensor([1, 0])
            else:
                sample['sex'] = torch.Tensor([0, 1])
            MINIMUM_AGE = 20
            MAXIMUM_AGE = 100
            
               
            norm_age_ct = (row['age'] - MINIMUM_AGE) / (MAXIMUM_AGE - MINIMUM_AGE)
            sample['age_ct'] = torch.Tensor([norm_age_ct])
                                                
        return sample

    
    # @property
    # def CORRUPTED_PATHS(self):
    #     return pickle.load(open(CORRUPTED_PATHS, "rb"))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        try:
            item = {}
            out_dict = self.get_images(sample)
            x = out_dict["input"]

            # if self.args.use_risk_factors:
            #     item["risk_factors"] = sample["risk_factors"]

            item["x"] = x.float()
            item['2x'] = out_dict['vis']
           
            if 'plus' in self.args.model_type:
                clinical_dict = {}
                clinical_dict['age_ct'] = sample['age_ct']
                clinical_dict['sex'] = sample['sex']
                item['x'] = {'img': item['x'], 'clinical': clinical_dict}
                item['x']['puid'] = sample['puid']
                item['x']['sex_str'] = '-1'
            # for key in CT_ITEM_KEYS:
            #     if key in sample:
            #         item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["puid"], traceback.print_exc()))

    def get_images(self, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        
        
        dicom_dataset = dicom_loader.load_dicom(sample['dir_path'])
      
        dicom_data =  list(dicom_dataset.values())[0]
        dicom_pixels = dicom_data['dicom_pixels'] 

        resampled, _ = resize(dicom_pixels, dicom_data['pixel_spacing'], [1,1,1], 'nearest')
        resampled2, _ = resize(dicom_pixels, dicom_data['pixel_spacing'], [1,0.5,0.5], 'nearest')
        resampled = np.rint(resampled).astype(np.int16)
        resampled2 = np.rint(resampled2).astype(np.int16)


        
        windowing=[1500, -600]
        W, L = windowing
        for series_uid, contents  in dicom_dataset.items():
            images = resampled # 512, 512, 336
            images2 = resampled2 # 512, 512, 336
            lowerb = L - W//2
            upperb = L + W//2
            images[images < lowerb] = lowerb
            images[images > upperb] = upperb
            images = images - lowerb
            images = np.float32(images * (255 / W))
            images2[images2 < lowerb] = lowerb
            images2[images2 > upperb] = upperb
            images2 = images2 - lowerb
            images2 = np.float32(images2 * (255 / W))
            sample['pixel_spacing'] = contents['pixel_spacing'].tolist()  # ndarray ->  list

        
        input_arr = self.reshape_images(images)  # 1, H, W, T 
        input_arr2 = self.reshape_images(images2)  # 1, H, W, T 
        # input_arr = self.resample_transform(input_arr)
        # input_arr = self.padding_transform(input_arr.data)

        input_arr = self.center_crop_transform(input_arr)
        input_arr2 = self.center_crop_transform2(input_arr2)
        # Convert from (C, T, H, W) to C, W, T, H
        input_arr = input_arr.repeat(3, 1, 1, 1) # TODO no need 3 ch. 
        input_arr2 = input_arr2.repeat(3, 1, 1, 1) # TODO no need 3 ch. 
        out_dict["input"] = input_arr.data/255
        out_dict['vis'] = input_arr2

        return out_dict

    def reshape_images(self, images):
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, axis=0) # 1, H, W, T = 1, 512, 512, 336
        # Convert from (C, H, W, T) to (C, T, H, W) # is it converted? It doesn't seem to be converted
        return images



# class NLST_Risk_Factor_Task(NLST_Survival_Dataset):
#     """
#     Dataset for risk factor-based risk model
#     """

#     def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
#         return self.risk_factor_vectorizer.get_risk_factors_for_sample(
#             pt_metadata, screen_timepoint
#         )
