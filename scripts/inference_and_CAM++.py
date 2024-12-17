import os
import sys 
sys.path.append('../')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from glob import glob
import warnings
from tqdm.auto import tqdm
from sybil.datasets.snuh_dicom_inference_2x import Inference_Dataset
from sybil.datasets.snuh_h5 import SNUH_Survival_Dataset
from argparse import Namespace
import numpy as np
import PIL.Image
import PIL.ImageDraw
from IPython import display
from ipywidgets import interact
import pdb
from sybil.utils import loading
from sybil.utils.helpers import get_dataset
from sybil.models.sybil import SybilNetPlusVolumeSurvivalDetachCAM, SybilNetPlusVolumeSurvivalDetachDoubleDensityCAM
from sybil.models.sybil import SybilNetPlusVolumeSurvivalDetach, SybilNetPlusVolumeSurvivalDetachDoubleDensity
import torch
import pandas as pd
from tqdm import trange
from scipy.ndimage import zoom
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
import argparse
import pydicom
from datetime import datetime

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.ioff()

def cudify(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = cudify(v)
        else:
            if not isinstance(v, list):
                d[k] = d[k].cuda().half()
    return d

def norm(att):
    att = (att - np.min(att)) / (np.max(att) - np.min(att))
    return att

def inference(df, save_dir):
    #label

    test_args = Namespace(train=False, test=True, dev=False, fine_tune=False, num_epochs_fine_tune=1, dataset='snuh_h5', img_size=[256, 256], num_chan=3, img_mean=[128.1722], img_std=[87.1849], img_dir='./', img_file_type='png', fix_seed_for_multi_image_augmentations=True, dataset_file_path='./', num_classes=6, cross_val_seed=0, assign_splits=False, split_type='random', split_probs=[0.6, 0.2, 0.2], max_followup=6, use_risk_factors=False, risk_factor_keys=[], resample_pixel_spacing_prob=1.0, num_images=300, min_num_images=0, slice_thickness_filter=2.5, use_only_thin_cuts_for_ct=True, use_annotations=True, region_annotations_filepath=None, annotation_loss_lambda=1, image_attention_loss_lambda=1, volume_attention_loss_lambda=1, primary_loss_lambda=1.0, adv_loss_lambda=1.0, batch_size=1, init_lr=1e-05, dropout=0.1, optimizer='adam', momentum=0.9, lr_decay=0.1, weight_decay=0.001, adv_lr=0.001, patience=5, num_adv_steps=1, tuning_metric='c_index', turn_off_checkpointing=False, save_dir='/data1/IPF_CT/sybil.ckpts/save_test_32_cache', snapshot=None, num_workers=4, store_hiddens=False, save_predictions=False, hiddens_dir='hiddens/test_run', save_attention_scores=False, results_path='/data1/IPF_CT/sybil.results_32_cache/results_test_32', cache_path='/data1/IPF_CT/cache/cache', cache_full_img=True, checkpoint_callback=None, enable_checkpointing=True, default_root_dir=None, gradient_clip_val=None, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=1, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=None, max_epochs=200, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, limit_predict_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=None, log_every_n_steps=50, accelerator='gpu', strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=0, resume_from_checkpoint=None, profiler='simple', benchmark=False, deterministic=False, reload_dataloaders_every_n_epochs=0, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=False, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None, lr=1e-05, unix_username='user', step_indx=1, callbacks=None, world_size=1, global_rank=0, local_rank=0, censoring_distribution={'0': 0.879282218597063, '1': 0.7744888070696665, '2': 0.6838571381572583, '3': 0.6229902724573978, '4': 0.5445867347671808, '5': 0.4268382515742769}, slice_thickness=1,  random_crop=False,random_crop_sigma=0.1, model_type='sybil_plus_volume_survival_detach', pred_volume=['fibrosis_proportion', 'normal_proportion'], mask_type='softmax', pool_type=['1','2','3'], pixel_spacing='111', mask_name=['honeycomb_reticular'])
    ds = Inference_Dataset(test_args, df)

    train_args = Namespace(train=True, test=True, dev=False, fine_tune=False, num_epochs_fine_tune=1, dataset='snuh_h5', img_size=[256, 256], num_chan=3, img_mean=[128.1722], img_std=[87.1849], img_dir='./', img_file_type='png', fix_seed_for_multi_image_augmentations=True, dataset_file_path='./', num_classes=6, cross_val_seed=0, assign_splits=False, split_type='random', split_probs=[0.6, 0.2, 0.2], max_followup=6, use_risk_factors=False, risk_factor_keys=[], resample_pixel_spacing_prob=1.0, num_images=300, min_num_images=0, slice_thickness_filter=2.9, use_only_thin_cuts_for_ct=True, use_annotations=True, region_annotations_filepath=None, annotation_loss_lambda=1, image_attention_loss_lambda=1, volume_attention_loss_lambda=1, primary_loss_lambda=1.0, adv_loss_lambda=1.0, batch_size=1, init_lr=1e-05, dropout=0.1, optimizer='adam', momentum=0.9, lr_decay=0.1, weight_decay=0.001, adv_lr=0.001, patience=5, num_adv_steps=1, tuning_metric='c_index', turn_off_checkpointing=False, save_dir='/data1/IPF_CT/sybil.ckpts/save_test_32_cache', snapshot=None, num_workers=4, store_hiddens=False, save_predictions=False, hiddens_dir='hiddens/test_run', save_attention_scores=False, results_path='/data1/IPF_CT/sybil.results_32_cache/results_test_32', cache_path='/data1/IPF_CT/cache/cache', cache_full_img=True, checkpoint_callback=None, enable_checkpointing=True, default_root_dir=None, gradient_clip_val=None, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=1, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=None, max_epochs=200, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, limit_predict_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=None, log_every_n_steps=50, accelerator='gpu', strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=0, resume_from_checkpoint=None, profiler='simple', benchmark=False, deterministic=False, reload_dataloaders_every_n_epochs=0, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=False, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None, lr=1e-05, unix_username='user', step_indx=1, callbacks=None, world_size=1, global_rank=0, local_rank=0, censoring_distribution={'0': 0.879282218597063, '1': 0.7744888070696665, '2': 0.6838571381572583, '3': 0.6229902724573978, '4': 0.5445867347671808, '5': 0.4268382515742769}, slice_thickness=1, random_crop=False, random_crop_sigma=0.1, model_type='sybil_plus_volume_survival_detach', pred_volume=['fibrosis_proportion', 'normal_proportion'], mask_type='softmax', pool_type=['1','2','3'], pixel_spacing='111', mask_name=['honeycomb_reticular'])
    val_args = Namespace(train=False, test=False, dev=True, fine_tune=False, num_epochs_fine_tune=1, dataset='snuh_h5', img_size=[256, 256], num_chan=3, img_mean=[128.1722], img_std=[87.1849], img_dir='./', img_file_type='png', fix_seed_for_multi_image_augmentations=True, dataset_file_path='./', num_classes=6, cross_val_seed=0, assign_splits=False, split_type='random', split_probs=[0.6, 0.2, 0.2], max_followup=6, use_risk_factors=False, risk_factor_keys=[], resample_pixel_spacing_prob=1.0, num_images=300, min_num_images=0, slice_thickness_filter=2.9, use_only_thin_cuts_for_ct=True, use_annotations=True, region_annotations_filepath=None, annotation_loss_lambda=1, image_attention_loss_lambda=1, volume_attention_loss_lambda=1, primary_loss_lambda=1.0, adv_loss_lambda=1.0, batch_size=1, init_lr=1e-05, dropout=0.1, optimizer='adam', momentum=0.9, lr_decay=0.1, weight_decay=0.001, adv_lr=0.001, patience=5, num_adv_steps=1, tuning_metric='c_index', turn_off_checkpointing=False, save_dir='/data1/IPF_CT/sybil.ckpts/save_test_32_cache', snapshot=None, num_workers=4, store_hiddens=False, save_predictions=False, hiddens_dir='hiddens/test_run', save_attention_scores=False, results_path='/data1/IPF_CT/sybil.results_32_cache/results_test_32', cache_path='/data1/IPF_CT/cache/cache', cache_full_img=True, checkpoint_callback=None, enable_checkpointing=True, default_root_dir=None, gradient_clip_val=None, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=1, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=None, max_epochs=200, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, limit_predict_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=None, log_every_n_steps=50, accelerator='gpu', strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=0, resume_from_checkpoint=None, profiler='simple', benchmark=False, deterministic=False, reload_dataloaders_every_n_epochs=0, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=False, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None, lr=1e-05, unix_username='user', step_indx=1, callbacks=None, world_size=1, global_rank=0, local_rank=0, censoring_distribution={'0': 0.879282218597063, '1': 0.7744888070696665, '2': 0.6838571381572583, '3': 0.6229902724573978, '4': 0.5445867347671808, '5': 0.4268382515742769}, slice_thickness=1,  random_crop=False, random_crop_sigma=0.1, model_type='sybil_plus_volume_survival_detach', pred_volume=['fibrosis_proportion', 'normal_proportion'], mask_type='softmax', pool_type=['1','2','3'], pixel_spacing='111', mask_name=['honeycomb_reticular'])
    test_args = Namespace(train=False, test=True, dev=False, fine_tune=False, num_epochs_fine_tune=1, dataset='snuh_h5', img_size=[256, 256], num_chan=3, img_mean=[128.1722], img_std=[87.1849], img_dir='./', img_file_type='png', fix_seed_for_multi_image_augmentations=True, dataset_file_path='./', num_classes=6, cross_val_seed=0, assign_splits=False, split_type='random', split_probs=[0.6, 0.2, 0.2], max_followup=6, use_risk_factors=False, risk_factor_keys=[], resample_pixel_spacing_prob=1.0, num_images=300, min_num_images=0, slice_thickness_filter=2.9, use_only_thin_cuts_for_ct=True, use_annotations=True, region_annotations_filepath=None, annotation_loss_lambda=1, image_attention_loss_lambda=1, volume_attention_loss_lambda=1, primary_loss_lambda=1.0, adv_loss_lambda=1.0, batch_size=1, init_lr=1e-05, dropout=0.1, optimizer='adam', momentum=0.9, lr_decay=0.1, weight_decay=0.001, adv_lr=0.001, patience=5, num_adv_steps=1, tuning_metric='c_index', turn_off_checkpointing=False, save_dir='/data1/IPF_CT/sybil.ckpts/save_test_32_cache', snapshot=None, num_workers=4, store_hiddens=False, save_predictions=False, hiddens_dir='hiddens/test_run', save_attention_scores=False, results_path='/data1/IPF_CT/sybil.results_32_cache/results_test_32', cache_path='/data1/IPF_CT/cache/cache', cache_full_img=True, checkpoint_callback=None, enable_checkpointing=True, default_root_dir=None, gradient_clip_val=None, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=1, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=None, max_epochs=200, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, limit_predict_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=None, log_every_n_steps=50, accelerator='gpu', strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=0, resume_from_checkpoint=None, profiler='simple', benchmark=False, deterministic=False, reload_dataloaders_every_n_epochs=0, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=False, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None, lr=1e-05, unix_username='user', step_indx=1, callbacks=None, world_size=1, global_rank=0, local_rank=0, censoring_distribution={'0': 0.879282218597063, '1': 0.7744888070696665, '2': 0.6838571381572583, '3': 0.6229902724573978, '4': 0.5445867347671808, '5': 0.4268382515742769}, slice_thickness=1,  random_crop=False,random_crop_sigma=0.1, model_type='sybil_plus_volume_survival_detach', pred_volume=['fibrosis_proportion', 'normal_proportion'], mask_type='softmax', pool_type=['1','2','3'], pixel_spacing='111', mask_name=['honeycomb_reticular'])


    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )


    # Construct the CAM object once, and then re-use it on many images.

    ckpts = [
        '/data1/IPF_CT/sybil.ckpts/1502_oc-adamw-111_softmax-bs4-lr1e-4-tmult1-t050-Tg1-256_256_300-E500-sybil_plus_volume_survival_detach_double_density-cropped_normal_densitycropped_fibrosis_density-r1-1502_oc/c_index/epoch=217-step=8065.ckpt',
        '/data1/IPF_CT/sybil.ckpts/1204-adamw-111_softmax-bs4-lr1e-4-tmult1-t050-Tg1-256_256_300-E200-sybil_plus_volume_survival_detach_double_density-cropped_normal_densitycropped_fibrosis_density-1204/c_index/epoch=120-step=9316.ckpt',
        '/data1/IPF_CT/sybil.ckpts/1200-adamw-111_softmax-bs4-lr1e-4-tmult1-t050-Tg1-256_256_300-E200-sybil_plus_volume_survival_detach-cropped_normal_densitycropped_fibrosis_density-1200/c_index/epoch=113-step=8777.ckpt',
        '/data1/IPF_CT/sybil.ckpts/1401_re-adamw-111_softmax-bs4-lr1e-4-tmult1-t050-Tg1-256_256_300-E1200-sybil_plus_volume_survival_detach_double_density-cropped_normal_densitycropped_fibrosis_density-r1-1401_re/c_index/epoch=207-step=7695.ckpt',
        '/data1/IPF_CT/sybil.ckpts/1302_oc-adamw-111_softmax-bs4-lr1e-4-tmult1-t050-Tg1-256_256_300-E1200-sybil_plus_volume_survival_detach_double_density-cropped_normal_densitycropped_fibrosis_density-r1-1302_oc/c_index/epoch=316-step=12362.ckpt'
    ]
    cam = GradCAMPlusPlus
    cmap = cm.get_cmap('YlOrRd')

    # ckpt_to_model_for_probs = {}

    # puid_to_probs = {}
    # for batch in tqdm(data_loader, total=len(data_loader)):
    #     probs = []
    #     for ckpt in ckpts:
    #         if ckpt not in ckpt_to_model_for_probs:
    #             try:
    #                 model = SybilNetPlusVolumeSurvivalDetachDoubleDensity(val_args)
    #                 loaded = torch.load(ckpt, map_location='cpu')
    #                 model.load_state_dict({ k[6:]:v for k,v in loaded['state_dict'].items() }, strict=True)
    #             except Exception as e:
    #                 model = SybilNetPlusVolumeSurvivalDetach(val_args)
    #                 loaded = torch.load(ckpt, map_location='cpu')
    #                 model.load_state_dict({ k[6:]:v for k,v in loaded['state_dict'].items() }, strict=True)

    #             model.eval()
    #             model.cuda()
    #             model.half()
    #             ckpt_to_model_for_probs[ckpt] = model # model initialize
    #         else:
    #             model = ckpt_to_model_for_probs[ckpt] # select an activated model in GPU memory
        
    #         with torch.no_grad():
    #             res = model(cudify(batch['x']))
    #             prob = torch.sigmoid(res['logit'][0]).detach().cpu().numpy()
    #             probs.append(prob)
    #     probs = np.mean(np.stack(probs, axis=0), axis=0) # probs is 1d array length of 6
    #     puid_to_probs[batch['x']['puid'][0]] = probs
    
    # #puid_to_probs to csv, # probs is 1d array length of 6 ( 0~1yr, 1~2yr, 2~3yr, 3~4yr, 4~5yr, 5yr~)
    # df = pd.DataFrame.from_dict(puid_to_probs, orient='index', columns=['0~1yr', '1~2yr', '2~3yr', '3~4yr', '4~5yr', '5yr~'])
    # os.makedirs(save_dir, exist_ok=True)
    # df.to_csv(f'{save_dir}/prediction.csv')
    
    # print(f'risk prediction is saved in {save_dir}/prediction.csv')
    
    for batch in tqdm(data_loader, total=len(data_loader)):
        cam_dict = {} 
        cmap_dict = {}
        for target_output in ['logit_volume_fibrosis', 'logit_volume_normal', 'logit'] :
            ckpt_to_model = {}

            if target_output == 'logit': 
                targets = [RawScoresOutputTarget()]*6
            else:
                targets = [RawScoresOutputTarget()]
            
            batch = cudify(batch)
            clinical = batch['x']['clinical']
            input_tensor = cudify(batch)['x']['img'] # Create an input tensor image for your model..
            cam_results = []
            for ckpt in ckpts:
                if ckpt not in ckpt_to_model:
                    try:
                        model = SybilNetPlusVolumeSurvivalDetachDoubleDensityCAM(val_args, clinical, output_target=target_output)
                        loaded = torch.load(ckpt, map_location='cpu')
                        model.load_state_dict({ k[6:]:v for k,v in loaded['state_dict'].items() }, strict=True)
                    except Exception as e:
                        model = SybilNetPlusVolumeSurvivalDetachCAM(val_args, clinical, output_target=target_output)
                        loaded = torch.load(ckpt, map_location='cpu')
                        model.load_state_dict({ k[6:]:v for k,v in loaded['state_dict'].items() }, strict=True)
                        
                    model.eval()
                    model.cuda()
                    model.half()
                    ckpt_to_model[ckpt] = model # model initialize
                else:
                    model = ckpt_to_model[ckpt] # select an activated model in GPU memory
                target_layers = [model.image_encoder[-1][-1]]
                     
                
                with cam(model=model, target_layers=target_layers) as c:
                        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                    grayscale_cam = c(input_tensor=input_tensor, targets=targets)
                    cam_results.append(grayscale_cam)
                
            cam_result_mean = np.mean(np.stack(cam_results), axis=0) # average the CAM activations of 5 models
            
            # Normalize the CAM result
            grayscale_cam = cam_result_mean.reshape([38, 16, 16])
            resize_factors = np.array([300, 512, 512]) / np.array([38, 16, 16])
            resized_cam =  zoom(grayscale_cam.astype(np.float32), resize_factors, order=1)
            final_cam_result = norm(resized_cam)

            cmap_arr = cmap(final_cam_result)
            cmap_arr[...,-1] = final_cam_result # fill the alpha channel
            cmap_dict[target_output] = cmap_arr
            cam_dict[target_output] = final_cam_result

        pid = batch['x']['puid'][0]
        img = batch['2x'][0].cpu().numpy()
            
            
        img = img.transpose(1,2,3,0).astype(np.uint8)
        for idx in range(len(img)):
            _img = img[idx]
            _survival = cmap_dict['logit'][idx]
            _normal = cmap_dict['logit_volume_normal'][idx]
            _fibrosis = cmap_dict['logit_volume_fibrosis'][idx]
            fig, ax = plt.subplots(figsize=(25 ,5), dpi=300)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            ax.set_xticks([])  # x  
            ax.set_yticks([])  # y  
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.imshow(np.concatenate([_img]*(1 + len(cmap_dict)), axis=1), cmap='gray', interpolation='none')
            cbar = fig.colorbar(
                ax.imshow(np.concatenate([np.zeros([512,512, 4])] + [_fibrosis, _normal, _survival], axis=1), alpha=1.0, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1), ax=ax
            )
            
            cbar.ax.set_position([0.82,0.02, 0.02, 0.96])
            os.makedirs(f'{save_dir}/{pid}', exist_ok=True)
            fig_path = f'{save_dir}/{pid}/PID{pid}-{str(idx).zfill(3)}of{len(img)}.png'
            fig.savefig(fig_path, dpi=300, bbox_inches=None, format='png', pad_inches=0, transparent=False,pil_kwargs={'compression_level': 0})
            plt.close(fig)
            crop = cv2.imread(fig_path)[:,:-1100]
            assert cv2.imwrite(fig_path, crop, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        
     
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory of target images')
    
    args = parser.parse_args()
    save_dir = args.save_dir
  
    # 각 directory 에서 가장 첫번째 dicom 파일을 읽어서 환자의 성별과 나이를 추출한다.
    # target_dir 의 구조는 다음과 같다.
    '''
    target_dir
    ├── ID1
    │   ├── 0001.dcm
    │   ├── 0002.dcm
    │   └── ...
    ├── ID2
    │   ├── 0001.dcm
    │   ├── 0002.dcm
    │   └── ...
    └── ...
    '''
    
    # ID1, ID2, ... 는 각각 다른 환자의 CT scan 이다.
    
    # inference를 하기 위해서는 df가 필요하다. 
    # df는 다음과 같은 형태이다.
    '''
    [ID, age, sex, dir_path]
    '''
    
    # 따라서 target directory 에서 각 directory 의 dicom 파일들을 읽어서 df를 생성한다.
    def generate_df(target_dir):
        df = pd.DataFrame(columns=['ID', 'age', 'sex', 'dir_path'])
        id_list = []
        age_list = []
        sex_list = []
        dir_path_list = []
        for idx, dir_path in enumerate(glob(target_dir + '/*')):
            dcm_path = list(glob(dir_path + '/*.dcm'))[0]
            #만약, dicom 파일을 읽을 수 없다면, 해당 directory 는 제외한다.
            try:
                d = pydicom.dcmread(dcm_path, force=True)
                ID = os.path.basename(dir_path)
                try:
                    days = (datetime.strptime(d.AcquisitionDate, '%Y%m%d') - datetime.strptime(d.PatientBirthDate, '%Y%m%d')).days
                    age = days/365
                except:
                    try:
                        days = (datetime.strptime(d.StudyDate, '%Y%m%d') - datetime.strptime(d.PatientBirthDate, '%Y%m%d')).days
                        age = days/365
                    except:
                        age = float(d.PatientAge.strip('0').strip('Y').strip('y').strip('YR').strip('yr').strip('YRS').strip('yrs'))

                id_list.append(ID)
                age_list.append(age)
                sex_list.append(d.PatientSex)
                dir_path_list.append(dir_path)
                
            except:
                print(f'Error: {dir_path} has invalid dicom file')
                continue 
            
        df['ID'] = id_list
        df['age'] = age_list
        df['sex'] = sex_list
        df['dir_path'] = dir_path_list
            
        return df
    
    df = generate_df(args.target_dir)
    inference(df, save_dir)
    