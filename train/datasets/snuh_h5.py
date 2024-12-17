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


class SNUH_Survival_Dataset(data.Dataset):
    def __init__(self, args, split_group, df_path='/data1/IPF_CT/snuh_label.csv' ):
        """
        SNUH Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(SNUH_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self._num_images = args.num_images  # number of slices in each volume # 336
        self._max_followup = args.max_followup # in Sybil paper, max_followup = 6 yrs

        self.label_df = pd.read_csv(df_path)
        if split_group == 'dev':
            self.label_df = self.label_df[(self.label_df.dataset == 'valid')]
        else:
            self.label_df = self.label_df[(self.label_df.dataset == split_group)]
       
        if split_group == 'train':
            self.label_df = self.label_df[self.label_df.thickness_list <= args.slice_thickness_filter]
        self.input_loader = get_sample_loader(split_group, args)
        self.always_resample_pixel_spacing = split_group in ["dev", "test"]

        self.padding_transform = tio.transforms.CropOrPad(
            # target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
            target_shape=tuple([args.num_images] +  [512,512] ), padding_mode=0
                               #[200] + [256, 256] 
        )
        self.center_crop_transform = tio.transforms.CropOrPad(
            target_shape=tuple([args.num_images] + args.img_size), padding_mode=0
                               #[200] + [256, 256]
        )
        self.random_crop_transform = RandomCrop( target_shape=tuple([args.num_images] + args.img_size), sigma=args.random_crop_sigma)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return
        
        print(self.get_summary_statement(self.dataset, split_group))

        dist_key = "y"
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {
            label: weight_per_label / count for label, count in label_counts.items()
        }

        print("Class counts are: {}".format(label_counts))
        print("Label weights are {}".format(label_weights))
        self.weights = [label_weights[d[dist_key]] for d in self.dataset]

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
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
        PREPATH = '/data1/IPF_CT/data/IPF_CT/'
        dir_path =  PREPATH + row['dir_path']


        pseudo_label_path = PREPATH + row['dir_path'] 
        
        y, y_seq, y_mask, time_at_event, time_at_event_f = self.get_label(row)


        sample = {
            "dir_path": dir_path,
            "y": int(y),
            "time_at_event": time_at_event,
            "time_at_event_f": time_at_event_f,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "pid": row.ID,
            'mask_path': pseudo_label_path,
        }
        if 'plus' in self.args.model_type:
            if row['성별'] == 'male':
            # if row['성별'] in ['male', 'M', 'Male', 'm']:
                sample['sex'] = torch.Tensor([1, 0])
            else:
                sample['sex'] = torch.Tensor([0, 1])
            smoking = row['smoking']
            if smoking == 'ever':
                sample['smoking'] = torch.Tensor([1, 0, 0])
            elif smoking == 'never':
                sample['smoking'] = torch.Tensor([0, 1, 0])
            else:
                sample['smoking'] = torch.Tensor([0, 0, 1])
            MINIMUM_HEIGHT = 130
            MAXIMUM_HEIGHT = 200
            sample['height'] = torch.Tensor([(row['Height_1']  - MINIMUM_HEIGHT) / (MAXIMUM_HEIGHT - MINIMUM_HEIGHT)])
            
            MINIMUM_WEIGHT = 30
            MAXIMUM_WEIGHT = 150 
            sample['weight'] = torch.Tensor([(row['Weight_1'] - MINIMUM_WEIGHT) / (MAXIMUM_WEIGHT - MINIMUM_WEIGHT)])
           
            MINIMUM_BMI = 10
            MAXIMUM_BMI = 40
            bmi = sample['weight'] / ((sample['height']/100)**2)
            sample['bmi'] =  torch.Tensor([(bmi - MINIMUM_BMI) / (MAXIMUM_BMI - MINIMUM_BMI)])
            
            MINIMUM_AGE = 20
            MAXIMUM_AGE = 100
            norm_age_dx = (row['진단시 나이'] - MINIMUM_AGE) / (MAXIMUM_AGE - MINIMUM_AGE)
            sample['age_dx'] = torch.Tensor([norm_age_dx])
            
            # handle excel serial number error
            try: 
                birth_date = datetime.strptime(row['생년월일'], '%Y-%m-%d')
            except:
                excel_serial_number = int(row['생년월일'])
                base_date = datetime(1899, 12, 30)
                birth_date = base_date + timedelta(days=excel_serial_number)
               
            ct_date = datetime.strptime(row['CT date'], '%Y-%m-%d %H:%M:%S')
            age_ct = (ct_date - birth_date).days // 365
            norm_age_ct = (age_ct - MINIMUM_AGE) / (MAXIMUM_AGE - MINIMUM_AGE)
            sample['age_ct'] = torch.Tensor([norm_age_ct])
            
            sample['normal_volume'] = torch.Tensor([row['ILDTextureNormalVolume(cc)_WholeLung']])
            sample['reticular_volume'] = torch.Tensor([row['ILDTextureReticularVolume(cc)_WholeLung']])
            sample['honeycomb_volume'] = torch.Tensor([row['ILDTextureHoneycombVolume(cc)_WholeLung']])
            sample['fibrosis_volume'] = torch.Tensor([row['ILDTextureReticularVolume(cc)_WholeLung'] + row['ILDTextureHoneycombVolume(cc)_WholeLung']])
            sample['total_lung_volume'] = torch.Tensor([row['ILDTextureVolume(cc)_WholeLung']])
            
            for v in self.args.pred_volume:
                if v.replace('proportion', 'volume') in sample:
                    sample[v] = sample[v.replace('proportion', 'volume')] / sample['total_lung_volume']
                                                
        return sample

    def get_label(self, row):
        years_to_event = row['event_year']
        years_to_event_f = row['event_year_float']
        y = row['event']
        y_seq = np.zeros(self.args.max_followup)
        time_at_event = min(years_to_event - 1, self.args.max_followup - 1) # if event not ocurred, the column"years_to_event" is defined as the last following days from first seen

        if y:
            y_seq[time_at_event:] = 1
        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.args.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event, years_to_event_f

    
    # @property
    # def CORRUPTED_PATHS(self):
    #     return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed SNUH CT Cancer Risk {} dataset with {} h5, {} mask, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        masks = set([d["mask_path"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(masks), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        return statement

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        try:
            item = {}
            out_dict = self.get_images(sample)
            x = out_dict["input"]
            volume_annotations_list = []
            annotation_areas_list = []
            image_annotations_list = []
            has_annotation_list = []
            for mask in out_dict['mask']:
                mask_area = mask.sum(dim=(-1, -2))
                volume_annotations = mask_area[0] / max(1, mask_area.sum())
                annotation_areas = mask_area[0] / (
                    mask.shape[-2] * mask.shape[-1]
                )
                mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
                mask_area[mask_area == 0] = 1 # prevent divided by zero
                if self.args.mask_type == 'softmax':
                    image_annotations = mask / mask_area # TODO:  why this line applied density??
                else:
                    image_annotations = mask.float()
                # image_annotations = mask 
                has_annotation = volume_annotations.sum() > 0
                
                volume_annotations_list.append(volume_annotations)
                annotation_areas_list.append(annotation_areas)
                image_annotations_list.append(image_annotations)
                has_annotation_list.append(has_annotation)

            item["volume_annotations"] = volume_annotations_list
            item["annotation_areas"] = annotation_areas_list
            item["image_annotations"] = image_annotations_list
            item["has_annotation"] = has_annotation_list
            # if self.args.use_risk_factors:
            #     item["risk_factors"] = sample["risk_factors"]

            item["x"] = x.float()
            item["y"] = {}
            item['y']['survival'] = sample['y']
            item['y']['cropped_fibrosis_density'] = out_dict['fibrosis_density']
            item['y']['cropped_lung_density'] = out_dict['lung_density']
            item['y']['cropped_normal_density'] = out_dict['normal_density']
            item['y']['cropped_normal_proportion'] = (out_dict['normal_density'] /  out_dict['lung_density'])
            item['y']['cropped_fibrosis_proportion'] = out_dict['fibrosis_density'] /  out_dict['lung_density']
            item['y_seq'] = sample['y_seq']
            item['y_mask'] = sample['y_mask']
            item['time_at_event'] = sample['time_at_event']
            item['time_at_event_f'] = sample['time_at_event_f']
           
            if len(self.args.pred_volume) > 0:
                item['normal_volume'] = sample['normal_volume']
                item['reticular_volume'] = sample['reticular_volume']
                item['honeycomb_volume'] = sample['honeycomb_volume']
                item['fibrosis_volume'] = sample['fibrosis_volume']
                item['total_lung_volume'] = sample['total_lung_volume']
                
                for v in self.args.pred_volume:
                    if v in sample:
                        item['y'][v] = sample[v] # fibrosis_proportion 
            
            if 'plus' in self.args.model_type:
                clinical_dict = {}
                clinical_dict['age_ct'] = sample['age_ct']
                clinical_dict['sex'] = sample['sex']
                item['x'] = {'img': item['x'], 'clinical': clinical_dict, 'pid': sample['pid']}
                item['pid'] = sample['pid']
            # for key in CT_ITEM_KEYS:
            #     if key in sample:
            #         item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["pid"], traceback.print_exc()))

    def get_images(self, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        puid = '_'.join(sample['dir_path'].split('/')[-4:])
        
        if self.args.pixel_spacing == '111':
            h5_path = '/data1/ramdisk/IPF_CT/preprocessed_isotropic/' + puid + '.h5'
        else: 
            h5_path = '/data1/ramdisk/IPF_CT/preprocessed/' + puid + '.h5'
            
        with h5py.File(h5_path, 'r') as hf:
            images = hf[f'{self.args.pixel_spacing}_arr'][:]
            masks = hf[f'{self.args.pixel_spacing}_mask'][:]
            lungs = hf[f'{self.args.pixel_spacing}_lung'][:]
            normals = hf[f'{self.args.pixel_spacing}_normal'][:]
            
        windowing=[1500, -600]
        W, L = windowing
        lowerb = L - W//2
        upperb = L + W//2
        images[images < lowerb] = lowerb
        images[images > upperb] = upperb
        images = images - lowerb
        images = np.float32(images * (255 / W))
        sample['pixel_spacing'] = [1,0.65,0.65]  # ndarray ->  list

        input_arr = self.reshape_images(images)  # 1, H, W, T 
        mask_arr = self.reshape_images(masks)
        lung_arr = self.reshape_images(lungs)
        normal_arr = self.reshape_images(normals)
        

        mask_list = [] 
        lung_list = []
        normal_list = []
        if self.split_group != "train" or not self.args.random_crop:
            input_arr = self.center_crop_transform(input_arr)
            mask_arr = self.center_crop_transform(mask_arr) 
            lung_arr = self.center_crop_transform(lung_arr) 
            normal_arr = self.center_crop_transform(normal_arr) 
            mask_arr = mask_arr.data
            lung_arr = lung_arr.data
            normal_arr = normal_arr.data
            mask_list.append(mask_arr)
            lung_list.append(lung_arr)
            normal_list.append(normal_arr)
            
            fibrosis_density = (mask_list[0] > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            normal_density = (normal_list[0] > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            lung_density = (lung_list[0] > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            out_dict['fibrosis_density'] = fibrosis_density
            out_dict['lung_density'] = lung_density
            out_dict['normal_density'] = normal_density
        else:
            self.padding_transform = tio.transforms.CropOrPad(
                # target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
                target_shape=tuple([max(self.args.num_images, input_arr.shape[-1])] +  [max(input_arr.shape[-3], self.args.img_size[0] ), max(input_arr.shape[-2], self.args.img_size[1])] ), padding_mode=0
                                #  [200] + [256, 256]
            )
            input_arr = self.padding_transform(input_arr)
            mask_arr = self.padding_transform(mask_arr) 
            lung_arr = self.padding_transform(lung_arr) 
            normal_arr = self.padding_transform(normal_arr) 
            if isinstance(input_arr, tio.ScalarImage):
                input_arr = input_arr.data
            if isinstance(mask_arr, tio.ScalarImage):
                mask_arr = mask_arr.data
            if isinstance(lung_arr, tio.ScalarImage):
                lung_arr = lung_arr.data
            if isinstance(normal_arr, tio.ScalarImage):
                normal_arr = normal_arr.data
            subject = tio.Subject(
                    input=tio.ScalarImage(tensor=input_arr), 
                    fibrosis=tio.ScalarImage(tensor=mask_arr), 
                    lung=tio.ScalarImage(tensor=lung_arr),
                    normal=tio.ScalarImage(tensor=normal_arr),
                ) 
           
            cropped_subject = self.random_crop_transform(subject)
            input_arr = cropped_subject['input'].data
            mask_list = [cropped_subject['fibrosis'].data]
            lung_list = [cropped_subject['lung'].data]
            normal_list = [cropped_subject['normal'].data]
            # if self.args.mask_name == ['honeycomb_reticular']:
            fibrosis_density = (cropped_subject['fibrosis'].data > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            out_dict['fibrosis_density'] = fibrosis_density
            normal_density = (cropped_subject['normal'].data > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            out_dict['normal_density'] = normal_density
            lung_density = (cropped_subject['lung'].data > 0).sum() / (self.args.img_size[0] * self.args.img_size[1] * self.args.num_images)
            out_dict['lung_density'] = lung_density
            


        # Convert from (C, T, H, W) to C, W, T, H
        input_arr = input_arr.repeat(3, 1, 1, 1) # TODO no need 3 ch. 
        out_dict["input"] = input_arr.data/255
        if self.args.use_annotations:
            out_dict["mask"] = mask_list
            out_dict["lung"] = lung_list
            out_dict["normal"] = normal_list

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
