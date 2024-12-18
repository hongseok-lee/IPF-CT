from collections import OrderedDict
from argparse import Namespace
import pickle
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from train.utils.helpers import get_dataset
from train.utils import losses as losses
from train.utils import metrics as metrics
from train.utils import loading as loaders
from train.models import train as model
from train.parsing import parse_args
import numpy as np

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class SybilLightning(pl.LightningModule):
    """
    Lightning Module
    Methods:
        .log/.log_dict: log inputs to logger
    Notes:
        *_epoch_end method returns None
        self can log additional data structures to logger
        with self.logger.experiment.log_*
        (*= 'text', 'image', 'audio', 'confusion_matrix', 'histogram')
    """

    def __init__(self, args):
        super(SybilLightning, self).__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        if args.model_type == "sybil":
            self.model = model.SybilNet(args)
        elif args.model_type == "sybil_plus":
            self.model = model.SybilNetPlus(args)
        elif args.model_type == "sybil_volume":
            self.model = model.SybilNetVolume(args)
        elif args.model_type == "sybil_plus_volume":
            self.model = model.SybilNetPlusVolume(args)
        elif args.model_type == "sybil_plus_volume_survival":
            self.model = model.SybilNetPlusVolumeSurvival(args)
        elif args.model_type == "sybil_plus_volume_survival_mtl":
            self.model = model.SybilNetPlusVolumeSurvivalMTL(args)
        elif args.model_type == "sybil_plus_volume_survival_detach":
            self.model = model.SybilNetPlusVolumeSurvivalDetach(args)
        elif args.model_type == "sybil_plus_volume_survival_detach_double_density":
            self.model = model.SybilNetPlusVolumeSurvivalDetachDoubleDensity(args)
           
        if args.load_pretrain: 
            loaded = torch.load(args.load_pretrain_path, map_location='cpu')
            state_dict = { k[6:]:v for k,v in loaded['state_dict'].items() }
            self.model.load_state_dict(state_dict, strict=False) 
       
        if args.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        
        self.save_prefix = "default"
        self.save_hyperparameters(args)
        if 'volume' in args.model_type and 'survival' not in args.model_type:
            self._list_of_metrics = [
                metrics.get_regression_metrics,
            ]
        elif 'volume' in args.model_type and 'survival' in args.model_type:
            self._list_of_metrics = [
                metrics.get_regression_metrics,
                metrics.get_survival_metrics,
            ]
        else: 
            self._list_of_metrics = [
                metrics.get_survival_metrics,
            ]

    def set_finetune(self, finetune_flag):
        return

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, optimizer_idx, log_key_prefix=""):
        model_output = self(batch["x"])
        logging_dict, predictions_dict = OrderedDict(), OrderedDict()
        predictions_dict['probs'] = {}
        predictions_dict['golds'] = {}

        if "exam" in batch:
            predictions_dict["exam"] = batch["exam"]
        if "y" in batch:
            predictions_dict["golds"] = batch["y"]
        if 'time_at_event_f' in batch:
            predictions_dict["time_at_event_f"] = batch["time_at_event_f"]

        if self.args.save_attention_scores:
            attentions = {k: v for k, v in model_output.items() if "attention" in k}
            predictions_dict.update(attentions)

        loss_fns = self.get_loss_functions(self.args)
        loss = 0
        for loss_fn in loss_fns:
            local_loss, local_log_dict, local_predictions_dict = loss_fn(
                model_output, batch, self, self.args
            )
            loss += local_loss
            logging_dict.update(local_log_dict)
            if "sybil_plus_volume_survival" in self.args.model_type or self.args.use_weak_loss:
                if 'probs' in local_predictions_dict and isinstance(predictions_dict['probs'], dict):
                    predictions_dict['probs'].update(local_predictions_dict['probs'])                
                if 'golds' in local_predictions_dict and isinstance(predictions_dict['golds'], dict):
                    predictions_dict['golds'].update(local_predictions_dict['golds'])
                if 'censors' in local_predictions_dict:
                    predictions_dict['censors']= local_predictions_dict['censors']
                # for k in local_predictions_dict.keys():
                #     if k not in ['probs', 'golds']:
                #         if k not in predictions_dict:
                #             predictions_dict[k] = {}
                #         predictions_dict[k].update(local_predictions_dict[k])
                    
            else:
                predictions_dict.update(local_predictions_dict)
        logging_dict = prefix_dict(logging_dict, log_key_prefix)
        predictions_dict = prefix_dict(predictions_dict, log_key_prefix)
        return loss, logging_dict, predictions_dict, model_output

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, _ = self.step(
            batch, batch_idx, optimizer_idx, log_key_prefix="train_"
        )
        logging_dict["train_loss"] = loss.detach()

        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        logging_dict['LR'] = lr
        self.log_dict(logging_dict, prog_bar=False, on_step=True, on_epoch=True)
        result["logs"] = logging_dict
        self.log_tensor_dict(predictions_dict, prog_bar=False, logger=False)
        result.update(predictions_dict)
        # lightning expects 'loss' key in output dict. ow loss := None by default
        result["loss"] = loss
        return result

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, _ = self.step(
            batch, batch_idx, optimizer_idx, log_key_prefix="val_"
        )
        logging_dict["val_loss"] = loss.detach()
        self.log_dict(logging_dict, prog_bar=True, sync_dist=True)
        result["logs"] = logging_dict
        if "ddp" in self.args.accelerator:
            predictions_dict = gather_predictions_dict(predictions_dict)
        self.log_tensor_dict(predictions_dict, prog_bar=False, logger=False)
        result.update(predictions_dict)
        return result

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, model_output = self.step(
            batch, batch_idx, optimizer_idx, log_key_prefix="test_"
        )
        logging_dict["{}_loss".format(self.save_prefix)] = loss.detach()
        result["logs"] = logging_dict

        if "ddp" in self.args.accelerator:
            predictions_dict = gather_predictions_dict(predictions_dict)

        self.log_tensor_dict(predictions_dict, prog_bar=False, logger=False)
        result.update(predictions_dict)
        return result

    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        # loss already logged in progress_bar_dict (get_progress_bar_dict()),
        # and logging twice creates issue
        del outputs["loss"]
        epoch_metrics = compute_epoch_metrics(
            self._list_of_metrics, outputs, self.args, self.device, key_prefix="train_"
        )
        for k, v in outputs["logs"].items():
            if isinstance(v, list):
                epoch_metrics[k] = np.mean(v)
            else:
                epoch_metrics[k] = v.mean()
        self.log_dict(epoch_metrics, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        epoch_metrics = compute_epoch_metrics(
            self._list_of_metrics, outputs, self.args, self.device, key_prefix="val_"
        )
        for k, v in outputs["logs"].items():
            if isinstance(v, list):
                epoch_metrics[k] = np.mean(v)
            else:
                epoch_metrics[k] = v.mean()
        self.log_dict(epoch_metrics, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        self.save_prefix= 'test'
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        epoch_metrics = compute_epoch_metrics(
            self._list_of_metrics, outputs, self.args, self.device, key_prefix="test_"
        )

        for k, v in outputs["logs"].items():
            if isinstance(v, list):
                epoch_metrics[k] = np.mean(v)
            else:
                epoch_metrics[k] = v.mean()

        self.log_dict(epoch_metrics, prog_bar=True, logger=True)

        # Dump metrics for use by dispatcher
        metrics_dict = {
            k[len(self.save_prefix) :]: v.mean().item()
            for k, v in outputs.items()
            if "loss" in k
        }
        metrics_dict.update(
            {
                k[len(self.save_prefix) :]: v.mean().item()
                for k, v in epoch_metrics.items()
            }
        )
        metrics_filename = "{}.{}.metrics".format(
            self.args.results_path, self.save_prefix
        )
        pickle.dump(metrics_dict, open(metrics_filename, "wb"))
        if self.args.save_predictions and self.global_rank == 0:
            predictions_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
            predictions_filename = "{}.{}.predictions".format(
                self.args.results_path, self.save_prefix
            )
            pickle.dump(predictions_dict, open(predictions_filename, "wb"))

    def configure_optimizers(self):
        """
        Helper function to fetch optimizer based on args.
        """
        params = [param for param in self.model.parameters() if param.requires_grad]
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params, lr=args.lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(
                params, lr=args.lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=args.lr, weight_decay=self.args.weight_decay
            )
        else:
            raise Exception("Optimizer {} not supported!".format(self.args.optimizer))


        steps_per_epoch = len(args.train_dataset)
        if self.args.scheduler == 'onecyclelr': 
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                                optimizer,
                                max_lr=self.args.lr,
                                steps_per_epoch=steps_per_epoch,
                                epochs=self.trainer.max_epochs,
                                pct_start=0.3,
                                anneal_strategy='cos',
                                div_factor=25,
                                final_div_factor=1e4
                ),
                'interval': 'step',
                }
        elif self.args.scheduler == 'reducelronplateau':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.args.patience,
                    factor=self.args.lr_decay,
                    mode="min" if "loss" in self.args.tuning_metric else "max",
                ),
                "monitor": "val_{}".format(self.args.tuning_metric),
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.args.scheduler == 'cosineannealing':
            
            if self.args.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    params, lr=1e-6, weight_decay=self.args.weight_decay
                )
            elif self.args.optimizer == "adagrad":
                optimizer = torch.optim.Adagrad(
                    params, lr=1e-6, weight_decay=self.args.weight_decay
                )
            elif self.args.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    params,
                    lr=1e-6,
                    weight_decay=self.args.weight_decay,
                    momentum=self.args.momentum,
                )
            elif self.args.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    params, lr=1e-6, weight_decay=self.args.weight_decay
                )
            else:
                raise Exception("Optimizer {} not supported!".format(self.args.optimizer))
                
            scheduler = {
                'scheduler': CosineAnnealingWarmUpRestarts(
                    optimizer, 
                    T_0=self.args.t_0, # 10,
                    T_up=2,
                    T_mult=self.args.t_mult,
                    eta_max=self.args.lr,
                    gamma=self.args.t_gamma, #0.8,
                ),
                'interval':'epoch',
            }
        else:
            raise NotImplementedError
        return [optimizer], [scheduler]

    def log_tensor_dict(
        self,
        output,
        prog_bar=False,
        logger=True,
        on_step=None,
        on_epoch=None,
        sync_dist=False,
    ):
        dict_of_tensors = {
            k: v.float() for k, v in output.items() if isinstance(v, torch.Tensor)
        }
        self.log_dict(
            dict_of_tensors,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
        )

    def get_loss_functions(self, args):
        if 'volume' in  args.model_type and 'survival' not in args.model_type:
            loss_fns = [losses.get_focal_loss,
                # losses.get_bce_with_logits_loss,
                        losses.get_mae_loss]
        elif 'volume' in  args.model_type and 'survival' in args.model_type:
            loss_fns = [losses.get_focal_loss,
                # losses.get_bce_with_logits_loss,
                        losses.get_mae_loss_survival,
                        losses.get_survival_loss_survival]
        else:
            loss_fns = [losses.get_survival_loss,
                        ]
        if args.use_annotations:
            loss_fns.append(losses.get_annotation_loss)
        if args.use_weak_loss:
            loss_fns.append(losses.get_annotation_weak_loss)

        return loss_fns


def prefix_dict(d, prefix):
    r = OrderedDict()
    for k, v in d.items():
        r[prefix + k] = v
    return r


def gather_predictions_dict(predictions):
    gathered_preds = {}
    for k, v in predictions. items(): 
        if isinstance(v, dict) :
            gathered_preds[k] = gather_predictions_dict(v)
        elif isinstance(v, torch.Tensor):
            gathered_preds[k] = concat_all_gather(v)
        else:
            gathered_preds[k] = v
    return gathered_preds


def gather_step_outputs(outputs):
    output_dict = OrderedDict()
    if isinstance(outputs[-1], list):
        outputs = outputs[0]

    for k in outputs[-1].keys():
        if k == "logs":
            output_dict[k] = gather_step_outputs([output["logs"] for output in outputs])
        elif (
            isinstance(outputs[-1][k], torch.Tensor) and len(outputs[-1][k].shape) == 0
        ):
            output_dict[k] = torch.stack([output[k] for output in outputs])
        elif isinstance(outputs[-1][k], torch.Tensor):
            output_dict[k] = torch.cat([output[k] for output in outputs], dim=0)
        elif isinstance(outputs[-1][k], dict):
            output_dict[k] = gather_step_outputs([output[k] for output in outputs])
        else:
            output_dict[k] = [output[k] for output in outputs]
    return output_dict


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def compute_epoch_metrics(list_of_metrics, result_dict, args, device, key_prefix=""):
    stats_dict = OrderedDict()

    """
        Remove prefix from keys. For instance, convert:
        val_probs -> probs for standard handling in the metric fucntions
    """
    result_dict_wo_key_prefix = {}

    for k, v in result_dict.items():
        if isinstance(v, list) and isinstance(v[-1], torch.Tensor):
            v = torch.cat(v, dim=-1)
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if k == "meta":
            continue
        if key_prefix != "" and k.startswith(key_prefix):
            k_wo_prefix = k[len(key_prefix) :]
            result_dict_wo_key_prefix[k_wo_prefix] = v
        else:
            result_dict_wo_key_prefix[k] = v

    for k, v in result_dict["logs"].items():
        if k.startswith(key_prefix):
            result_dict_wo_key_prefix[k[len(key_prefix) :]] = v

    for metric_func in list_of_metrics:
        stats_wo_prefix = metric_func(result_dict_wo_key_prefix, args)
        for k, v in stats_wo_prefix.items():
            stats_dict[key_prefix + k] = torch.tensor(v, device=device)

    return stats_dict


def train(args):
    if not args.turn_off_checkpointing:
        args.callbacks = []
        for tm in args.tuning_metric:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=args.save_dir + '/' + tm,
                save_top_k=10,
                verbose=True,
                monitor="val_{}".format(tm)
                if tm is not None
                else None,
                save_last=True,
                mode="min"
                if tm is not None and ("loss" in tm.lower() or 'mae' in tm.lower() or 'mse' in tm.lower())
                else "max",
            )
            args.callbacks += [checkpoint_callback]
        args.logger = [
                        MLFlowLogger(experiment_name="default", tracking_uri="http://172.23.100.25:5000"), 
                        pl.loggers.TensorBoardLogger('/data1/IPF_CT/tensorboard_logs', name='Sybil-IPF', version=args.save_dir.split('/')[-1])
                    ]



    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank


    train_dataset = loaders.get_train_dataset_loader(
        args, get_dataset(args.dataset, "train", args)
    )
    dev_dataset = loaders.get_eval_dataset_loader(
        args, get_dataset(args.dataset, "dev", args), False
    )

    # train_dataset = loaders.get_train_dataset_loader(
    #     args, get_dataset(args.dataset, "train", args)
    # )
    # dev_dataset = loaders.get_eval_dataset_loader(
    #     args, get_dataset(args.dataset, "dev", args), False
    # )

    args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)
    if os.path.isfile('survival_train.npy'):
        args.survival_train = np.load('survival_train.npy')
    else:
        args.survival_train = np.array( [ (x['y']['survival'].numpy(), x['time_at_event_f']) for x in train_dataset ] , dtype=[('death', bool), ('futime', float)] )
    args.train_dataset = train_dataset
    module = SybilLightning(args)

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    if args.snapshot is not None:
        module = module.load_from_checkpoint(checkpoint_path= args.snapshot, strict=False)
        module.args = args
    
    trainer.fit(module, train_dataset, dev_dataset)
    args.model_path = trainer.checkpoint_callback.best_model_path
    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path, "wb"))

def test(args):
    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    train_dataset = loaders.get_train_dataset_loader(
        args, get_dataset(args.dataset, "train", args)
    )
    test_dataset = loaders.get_eval_dataset_loader(
        args, get_dataset(args.dataset, "test", args), False
    )

    if os.path.isfile('survival_train.npy'):
        args.survival_train = np.load('survival_train.npy')
    else:
        args.survival_train = np.array( [ (x['y']['survival'].numpy(), x['time_at_event_f']) for x in train_dataset ] , dtype=[('death', bool), ('futime', float)] )
    args.train_dataset = train_dataset
    
    args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)
    module = SybilLightning(args)
    module = module.load_from_checkpoint(checkpoint_path= args.snapshot, strict=False)
    module.args = args

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    trainer.test(module, test_dataset)

    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path, "wb"))

def valid(args):
    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    train_dataset = loaders.get_train_dataset_loader(
        args, get_dataset(args.dataset, "train", args)
    )
    test_dataset = loaders.get_eval_dataset_loader(
        args, get_dataset(args.dataset, "dev", args), False
    )
    
    if os.path.isfile('survival_train.npy'):
        args.survival_train = np.load('survival_train.npy')
    else:
        args.survival_train = np.array( [ (x['y']['survival'].numpy(), x['time_at_event_f']) for x in train_dataset ] , dtype=[('death', bool), ('futime', float)] )
    args.train_dataset = train_dataset

    args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)
    module = SybilLightning(args)
    module = module.load_from_checkpoint(checkpoint_path= args.snapshot, strict=False)
    module.args = args

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    trainer.test(module, test_dataset)

    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path, "wb"))



if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.dev:
        valid(args)
