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
import h5py


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
        
        for path_to_dcm in dicom_list:
            d = dicom.read_file(path_to_dcm, force=True)
            if not '0x00080016' in d: continue # super meta separator
            
            # ImageOrientationPatient 
            if '1.2.840.10008.5.1.4.1.1.2' in d.SOPClassUID\
                and np.sum(
                        np.array(d.ImageOrientationPatient, dtype=np.int8)\
                        - self.params.preffered_orientation) == 0:
                
                try:
                    dicom_paths[d[self.params.group_separator].value]\
                        += [path_to_dcm]
                    dicom_slices[d[self.params.group_separator].value]\
                        += [d]

                except:
                    dicom_paths[d[self.params.group_separator].value]\
                        = [path_to_dcm]
                    dicom_slices[d[self.params.group_separator].value]\
                        = [d]
            
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
                 for idx in [4, 8, -8, -4]]), decimals=1), return_counts=True) # 균일한 거리인지 검사. 4,8인거는 012는 잡영상 가능성 있음. 
            if len(spacing_between_slices) > 1 or max(spacing_between_slices) <= 1e-8:
                err_code += 1

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
"""

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


class SNUH_Survival_Dataset(data.Dataset):
    def __init__(self, args, split_group):
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

        self.label_df = pd.read_csv('/data1/IPF_CT/snuh_label.csv')
        if split_group == 'dev':
            self.label_df = self.label_df[(self.label_df.dataset == 'valid')]
        else:
            self.label_df = self.label_df[(self.label_df.dataset == split_group)]

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
        # self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        # self.corrupted_series = self.CORRUPTED_PATHS["series"]
        # self.risk_factor_vectorizer = NLSTRiskFactorVectorizer(self.args)

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
        pseudo_label_path = PREPATH + row['dir_path'] + '/masks.npz'
        
        y, y_seq, y_mask, time_at_event = self.get_label(row)


        sample = {
            "dir_path": dir_path,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "pid": row.ID,
            'mask_path': pseudo_label_path,
            #"cancer_laterality": self.get_cancer_side(pt_metadata), # should be decided. 
        }

        # if self.args.use_risk_factors:
        #     sample["risk_factors"] = self.get_risk_factors(
        #         pt_metadata, screen_timepoint, return_dict=False
        #     )

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, row):
        
        years_to_event = row['event_year']
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
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer
    
    

    def get_thinnest_cut(self, exam_dict):
        # volume that is not thin cut might be the one annotated; or there are multiple volumes with same num slices, so:
        # use annotated if available, otherwise use thinnest cut
        possibly_annotated_series = [
            s in self.annotations_metadata
            for s in list(exam_dict["image_series"].keys())
        ]
        series_lengths = [
            len(exam_dict["image_series"][series_id]["paths"])
            for series_id in exam_dict["image_series"].keys()
        ]
        thinnest_series_len = max(series_lengths)
        thinnest_series_id = [
            k
            for k, v in exam_dict["image_series"].items()
            if len(v["paths"]) == thinnest_series_len
        ] 
        if any(possibly_annotated_series):
            thinnest_series_id = list(exam_dict["image_series"].keys())[
                possibly_annotated_series.index(1)  # [HS] this is hard-coded selection of annotated_series. It can be advanced in aspect of augmentation. 
            ]
        else:
            thinnest_series_id = thinnest_series_id[0] # [HS] this is hard-coded selection of series. It can be advanced in aspect of augmentation. 
        return thinnest_series_id

    def skip_sample(self, series_dict, pt_metadata):
        series_data = series_dict["series_data"]
        # check if screen is localizer screen or not enough images
        is_localizer = self.is_localizer(series_data)

        # check if restricting to specific slice thicknesses
        slice_thickness = series_data["reconthickness"][0]
        wrong_thickness = (self.args.slice_thickness_filter is not None) and (
            slice_thickness > self.args.slice_thickness_filter or (slice_thickness < 0)
        )

        # check if valid label (info is not missing)
        screen_timepoint = series_data["study_yr"][0]
        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # invalid label
        if not bad_label:
            y, _, _, time_at_event = self.get_label(pt_metadata, screen_timepoint)
            invalid_label = (y == -1) or (time_at_event < 0)
        else:
            invalid_label = False

        insufficient_slices = len(series_dict["paths"]) < self.args.min_num_images

        if (
            is_localizer
            or wrong_thickness
            or bad_label
            or invalid_label
            or insufficient_slices
        ):
            return True
        else:
            return False


    def get_cancer_side(self, pt_metadata):
        """
        Return if cancer in left or right

        right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
        left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
        else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
        """
        right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
        left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
        other_keys = ["loccar", "locmed", "locoth", "locunk"]

        right = any([pt_metadata[key][0] > 0 for key in right_keys])
        left = any([pt_metadata[key][0] > 0 for key in left_keys])
        other = any([pt_metadata[key][0] > 0 for key in other_keys])

        return np.array([int(right), int(left), int(other)])

    def order_slices(self, img_paths, slice_locations):
        sorted_ids = np.argsort(slice_locations)
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        if not sorted_img_paths[0].startswith(self.args.img_dir):
            sorted_img_paths = [
                self.args.img_dir
                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                for path in sorted_img_paths
            ]
        if self.args.img_file_type == "dicom":
            sorted_img_paths = [
                path.replace("nlst-ct-png", "nlst-ct").replace(".png", "")
                for path in sorted_img_paths
            ]

        return sorted_img_paths, sorted_slice_locs

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        age_at_randomization = pt_metadata["age"][0]
        days_since_randomization = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        current_age = age_at_randomization + days_since_randomization // 365

        age_start_smoking = pt_metadata["smokeage"][0]
        age_quit_smoking = pt_metadata["age_quit"][0]
        years_smoking = pt_metadata["smokeyr"][0]
        is_smoker = pt_metadata["cigsmok"][0]

        years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        education = (
            pt_metadata["educat"][0]
            if pt_metadata["educat"][0] != -1
            else pt_metadata["educat"][0]
        )

        race = pt_metadata["race"][0] if pt_metadata["race"][0] != -1 else 0
        race = 6 if pt_metadata["ethnic"][0] == 1 else race
        ethnicity = pt_metadata["ethnic"][0]

        weight = pt_metadata["weight"][0] if pt_metadata["weight"][0] != -1 else 0
        height = pt_metadata["height"][0] if pt_metadata["height"][0] != -1 else 0
        bmi = weight / (height**2) * 703 if height > 0 else 0  # inches, lbs

        prior_cancer_keys = [
            "cancblad",
            "cancbrea",
            "canccerv",
            "canccolo",
            "cancesop",
            "canckidn",
            "canclary",
            "canclung",
            "cancoral",
            "cancnasa",
            "cancpanc",
            "cancphar",
            "cancstom",
            "cancthyr",
            "canctran",
        ]
        cancer_hx = any([pt_metadata[key][0] == 1 for key in prior_cancer_keys])
        family_hx = any(
            [pt_metadata[key][0] == 1 for key in pt_metadata if key.startswith("fam")]
        )

        risk_factors = {
            "age": current_age,
            "race": race,
            "race_name": RACE_ID_KEYS.get(pt_metadata["race"][0], "UNK"),
            "ethnicity": ethnicity,
            "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            "education": education,
            "bmi": bmi,
            "cancer_hx": cancer_hx,
            "family_lc_hx": family_hx,
            "copd": pt_metadata["diagcopd"][0],
            "is_smoker": is_smoker,
            "smoking_intensity": pt_metadata["smokeday"][0],
            "smoking_duration": pt_metadata["smokeyr"][0],
            "years_since_quit_smoking": years_since_quit_smoking,
            "weight": weight,
            "height": height,
            "gender": GENDER_KEYS.get(pt_metadata["gender"][0], "UNK"),
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def assign_splits(self, meta):
        if self.args.split_type == "institution_split":
            self.assign_institutions_splits(meta)
        elif self.args.split_type == "random": # default
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(
                    ["train", "dev", "test"], p=self.args.split_probs # 0.6 , 0.2, 0.2
                )

    def assign_institutions_splits(self, meta):
        institutions = set([m["pt_metadata"]["cen"][0] for m in meta])
        institutions = sorted(institutions)
        institute_to_split = {
            cen: np.random.choice(["train", "dev", "test"], p=self.args.split_probs)
            for cen in institutions
        }
        for idx in range(len(meta)):
            meta[idx]["split"] = institute_to_split[meta[idx]["pt_metadata"]["cen"][0]]

    # @property
    # def CORRUPTED_PATHS(self):
    #     return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} dicom, {} mask, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        masks = set([d["mask_path"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(masks), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        statement
        return statement

    @property
    def GOOGLE_SPLITS(self):
        return pickle.load(open(GOOGLE_SPLITS_FILENAME, "rb"))

    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        if sample["series"] in self.annotations_metadata:
            self.annotations_metadata[sample["series"]] = {
                k: v
                for k, v in self.annotations_metadata[sample["series"]].items()
                if len(v) > 0
            }

        if sample["series"] in self.annotations_metadata:
            # store annotation(s) data (x,y,width,height) for each slice
            if (
                self.args.img_file_type == "dicom"
            ):  # no file extension, so os.path.splitext breaks behavior
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.basename(path), None)
                    }
                    for path in sample["paths"]
                ]
            else:  # expects file extension to exist, so use os.path.splitext
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.splitext(os.path.basename(path))[0], None)
                    }
                    for path in sample["paths"]
                ]
        else:
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        try:
            item = {}
            out_dict = self.get_images(sample)
            x = out_dict["input"]
            mask = out_dict["mask"]

            mask_area = mask.sum(dim=(-1, -2))
            item["volume_annotations"] = mask_area[0] / max(1, mask_area.sum())
            item["annotation_areas"] = mask_area[0] / (
                mask.shape[-2] * mask.shape[-1]
            )
            mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
            mask_area[mask_area == 0] = 1
            item["image_annotations"] = mask / mask_area
            item["has_annotation"] = item["volume_annotations"].sum() > 0

            # if self.args.use_risk_factors:
            #     item["risk_factors"] = sample["risk_factors"]

            item["x"] = x
            item["y"] = sample["y"]
            item['y_seq'] = sample['y_seq']
            item['y_mask'] = sample['y_mask']
            item['time_at_event'] = sample['time_at_event']
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

        pid = sample['pid']
        target = f'/data1/IPF_CT/data/npy/{pid}.npy'
        npy_data = np.load(target, allow_pickle=True).item()
        pixel_spacing = list(npy_data.values())[0]['pixel_spacing']
        
        windowing=[1500, -600]
        W, L = windowing
        images = npy_data['pixel_array'] # 512, 512, 336
        lowerb = L - W//2
        upperb = L + W//2
        images[images < lowerb] = lowerb
        images[images > upperb] = upperb
        images = images - lowerb
        # images = np.uint8(images * (255 / W))
        images = np.array(images * (255 / W)) # user float
        sample['pixel_spacing'] = pixel_spacing.tolist()[0]

        input_arr = self.reshape_images(images) 
        
        mask_1 = np.load(sample['mask_path'])['honeycomb']
        # mask_2 = np.load(sample['mask_path'])['ggo'] # remove ggo to mask
        mask_3 = np.load(sample['mask_path'])['reticular']
        # mask = np.logical_or(mask_1, mask_2) # remove ggo to mask
        mask = np.logical_or(mask_1, mask_3)  # 512, 512, 336
        mask_arr = self.reshape_images(mask) if self.args.use_annotations else None

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

            if self.args.use_annotations:
                mask_arr = tio.ScalarImage(
                    affine=torch.diag(spacing),
                    tensor=mask_arr.permute(0, 2, 3, 1),
                )
                mask_arr = self.resample_transform(mask_arr)
                mask_arr = self.padding_transform(mask_arr.data)

        # Convert from (C, T, H, W) to C, W, T, H
        input_arr = input_arr.repeat(3, 1, 1, 1) # TODO no need 3 ch. 
        out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)/255
        if self.args.use_annotations:
            out_dict["mask"] = mask_arr.data.permute(0, 3, 1, 2)

        return out_dict

    def reshape_images(self, images):
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, axis=0) # 1, H, W, T = 1, 512, 512, 336
        
        
        # Convert from (C, H, W, T) to (C, T, H, W)
        return images

    def get_slice_thickness_class(self, thickness):
        BINS = [1, 1.5, 2, 2.5]
        for i, tau in enumerate(BINS):
            if thickness <= tau:
                return i
        if self.args.slice_thickness_filter is not None:
            raise ValueError("THICKNESS > 2.5")
        return 4


# class NLST_Risk_Factor_Task(NLST_Survival_Dataset):
#     """
#     Dataset for risk factor-based risk model
#     """

#     def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
#         return self.risk_factor_vectorizer.get_risk_factors_for_sample(
#             pt_metadata, screen_timepoint
#         )
