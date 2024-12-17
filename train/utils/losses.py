from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(inputs) for positive samples, 1-sigmoid(inputs) for negative samples
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def get_focal_loss(model_output, batch, model, args):
    criterion = FocalLoss(gamma=args.focal_loss_gamma)
    logging_dict, predictions = OrderedDict(), OrderedDict()
    loss = 0
    predictions["probs"] = {}
    predictions["golds"] = {}
    if len(args.pred_volume_lambda) > 0:
        lambdas = [float(x) for x in args.pred_volume_lambda]
    else:
        lambdas = [1] * len(args.pred_volume)
        
    for i, (v, _lambda) in enumerate(zip(sorted(args.pred_volume), lambdas)):
        logit = model_output["logit"][:, i]
        _loss = (criterion(logit, batch["y"][v].float().squeeze(-1)) * _lambda) 
        loss += _loss
        logging_dict[f'focal_loss_{v}'] = _loss.detach()
        pred = F.sigmoid(logit)
        predictions["probs"][v] = pred.detach()
        predictions["golds"][v] = batch["y"][v]
    return loss, logging_dict, predictions

def get_mae_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    loss = 0
    predictions["probs"] = {}
    predictions["golds"] = {}
    if len(args.pred_volume_lambda) > 0:
        lambdas = [float(x) for x in args.pred_volume_lambda]
    else:
        lambdas = [1] * len(args.pred_volume)
        
    for i, (v, _lambda) in enumerate(zip(sorted(args.pred_volume), lambdas)):
        logit = model_output["logit"][:, i]
        pred = F.sigmoid(logit)
        _loss = (F.l1_loss(pred, batch["y"][v].float().squeeze(-1)) * _lambda)
        loss += _loss
        logging_dict[f'MAE_loss_{v}'] = _loss.detach()
        predictions["probs"][v] = pred.detach()
        predictions["golds"][v] = batch["y"][v]
        
    return loss, logging_dict, predictions

def get_mse_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    loss = F.mse_loss(F.sigmoid(logit), batch["y"].float())
    logging_dict["MSE_loss"] = loss.detach()
    predictions["probs"] = logit.detach()
    predictions["golds"] = batch["y"]
    return loss, logging_dict, predictions

def get_mae_loss_survival(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    loss = 0
    predictions["probs"] = {}
    predictions["golds"] = {}
    for i, v in enumerate(sorted(args.pred_volume)):
        logit = model_output["logit_volume"][:, i]
        pred = F.sigmoid(logit)
        _loss = F.l1_loss(pred, batch["y"][v].float().squeeze(-1))
        loss += _loss
        logging_dict[f'MAE_loss_{v}'] = _loss.detach()
        predictions["probs"][v] = pred.detach()
        predictions["golds"][v] = batch["y"][v]
        
    return loss, logging_dict, predictions



def get_huber_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    loss = F.smooth_l1_loss(F.sigmoid(logit), batch["y"].float())
    logging_dict["Huber_loss"] = loss.detach()
    predictions["probs"] = logit.detach()
    predictions["golds"] = batch["y"]
    return loss, logging_dict, predictions

def get_bce_with_logits_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    predictions["probs"] = {}
    predictions["golds"] = {}
    loss = 0
    for i, v in enumerate(sorted(args.pred_volume)):
        logit = model_output["logit"][:,i]
        _loss = F.binary_cross_entropy_with_logits(logit, batch["y"][v].float().squeeze(-1))
        loss += _loss
        logging_dict[f'BCE_loss_{v}'] = _loss.detach()
        predictions['probs'][v] = F.sigmoid(logit).detach()
        predictions['golds'][v] = batch["y"][v]
    return loss, logging_dict, predictions

def get_bce_with_logits_loss_survival(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    predictions["probs"] = {}
    predictions["golds"] = {}
    loss = 0
    for i, v in enumerate(sorted(args.pred_volume)):
        logit = model_output["logit_volume"][:,i]
        _loss = F.binary_cross_entropy_with_logits(logit, batch["y"][v].float().squeeze(-1))
        loss += _loss
        logging_dict[f'BCE_loss_{v}'] = _loss.detach()
        predictions['probs'][v] = F.sigmoid(logit).detach()
        predictions['golds'][v] = batch["y"][v]
    return loss, logging_dict, predictions


def get_cross_entropy_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    loss = F.cross_entropy(logit, batch["y"].long())
    logging_dict["cross_entropy_loss"] = loss.detach()
    predictions["probs"] = F.softmax(logit, dim=-1).detach()
    predictions["golds"] = batch["y"]
    return loss, logging_dict, predictions


def get_survival_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    predictions["probs"] = {}
    predictions["golds"] = {}
    logit = model_output["logit"]
    y_seq, y_mask = batch["y_seq"], batch["y_mask"]
    loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    logging_dict["survival_loss"] = loss.detach()
    predictions["probs"]['survival'] = torch.sigmoid(logit).detach()
    predictions["golds"]['survival'] = batch["y"]['survival']
    predictions["censors"] = batch["time_at_event"]
    return loss, logging_dict, predictions

def get_survival_loss_survival(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    predictions["probs"] = {}
    predictions["golds"] = {}
    logit = model_output["logit"]
    y_seq, y_mask = batch["y_seq"], batch["y_mask"]
    loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    logging_dict["survival_loss"] = loss.detach()
    predictions["probs"]['survival'] = torch.sigmoid(logit).detach()
    predictions["golds"]['survival'] = batch["y"]['survival']
    predictions["censors"] = batch["time_at_event"]
    return loss, logging_dict, predictions

def get_annotation_loss_survival(model_output, batch, model, args):
    total_loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()

    B, _, N, H, W, = model_output["activ"].shape   # B, 3, 38, 24, 24

    if len(args.mask_name) == 1:
        batch["_image_annotations"] = batch['image_annotations'][0]
        batch_mask = batch["has_annotation"][0]
        batch["_annotation_areas"] = batch["annotation_areas"][0]

    for attn_num in [1, 2, 3]:
        if len(batch['image_annotations']) < attn_num:
            if model_output.get("image_attention_{}".format(attn_num), None) is not None: # in this branch, only image_attention_1 is available.
                batch["_image_annotations"] = batch['image_annotations'][0]
                batch_mask = batch["has_annotation"][0]
                batch["_annotation_areas"] = batch["annotation_areas"][0]
                
            else:
                continue
        if len(args.mask_name) == 2: # if mask is multiple, redefine batch
            batch["_image_annotations"] = batch['image_annotations'][attn_num-1]
            batch_mask = batch["has_annotation"][attn_num-1]
            batch["_annotation_areas"] = batch["annotation_areas"][attn_num-1]

        side_attn = -1
        if model_output.get("image_attention_{}".format(attn_num), None) is not None: # in this branch, only image_attention_1 is available.
                                                                                      # in Simple_AttentionPool_MultiImg
                                                                                      # image_attention -> B, T, WH (log of softmax output)
                
            if len(batch["_image_annotations"].shape) == 4:
                batch["_image_annotations"] = batch["_image_annotations"].unsqueeze(1) # convert into 5D tensor.  if channel is not exist, add it. 

            # resize annotation to 'activ' size
            annotation_gold = F.interpolate(
                batch["_image_annotations"], (N, H, W), mode=args.mask_interpolation
            )
            annotation_gold = annotation_gold * batch_mask[:, None, None, None, None] # interpolate되었더라도, batch 단위로 rule out. why??

            # renormalize scores
            mask_area = annotation_gold.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            mask_area[mask_area == 0] = 1 # depth wise mask sum
            if args.mask_type == 'softmax':
                annotation_gold /= mask_area
            elif args.mask_type == 'sigmoid':
                pass

            # reshape annotation into 1D vector
            annotation_gold = annotation_gold.view(B, N, -1).float()

            # get mask over annotation boxes in order to weigh
            # non-annotated scores with zero when computing loss
            annotation_gold_mask = (annotation_gold > 0).float() # B, N, WH

            num_annotated_samples = (annotation_gold.view(B * N, -1).sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples) # number of annotated slices

            if args.mask_type == 'softmax':
                pred_attn = (
                    model_output["image_attention_{}".format(attn_num)]  #B, T, WH  (B, 38, 576)
                    * batch_mask[:, None, None] # it is weird use batch_mask to predict attention. maybe it considering partial label??  #TODO remove it. 
                ) # TODO validate this block when mask_type is softmax
                kldiv = (
                    F.kl_div(pred_attn, annotation_gold, reduction="none")
                    * annotation_gold_mask # this multiply intend to allow that negative-annotated area is ignored. 어텐션 꼭 줘야하는것들은 loss 적용하고, no label이더라도 attention들어가는걸 막지는 않는것. 좀더 유연하다. 
                )

                # sum loss per volume and average over batches
                loss = kldiv.sum() / num_annotated_samples
            elif args.mask_type == 'sigmoid':
                pred_attn = model_output["image_attention_{}".format(attn_num)] 
                criterion = FocalLoss(gamma=args.focal_loss_gamma)
                # bce_criterion = nn.BCEWithLogitsLoss()
                loss = criterion(
                    pred_attn.view(B, N, -1),
                    annotation_gold
                )
            logging_dict["image_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.image_attention_loss_lambda * loss
           
        if model_output.get("volume_attention_{}".format(attn_num), None) is not None:
            # find size of annotation box per slice and normalize
            annotation_gold = batch["_annotation_areas"].float() * batch_mask[:, None] # TODO: area_per_slice may have been wrong from origimal repo, otherwise, it is misnaming. ckeck it. 
            if N != args.num_images:
                annotation_gold = F.interpolate(annotation_gold.unsqueeze(1), (N), mode= 'linear', align_corners = True)[:,0]
            if args.mask_type == 'softmax': 
                # Normalize area_per_slice on depth level
                area_per_slice = annotation_gold.sum(-1).unsqueeze(-1)
                area_per_slice[area_per_slice == 0] = 1
                annotation_gold /= area_per_slice
            elif args.mask_type == 'sigmoid':
                pass
            num_annotated_samples = (annotation_gold.sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples)

            # find slices with annotation
            annotation_gold_mask = (annotation_gold > 0).float()

            pred_attn = (
                model_output["volume_attention_{}".format(attn_num)]
                * batch_mask[:, None]
            )

            if args.mask_type == 'softmax':
                kldiv = (
                    F.kl_div(pred_attn, annotation_gold, reduction="none")
                    * annotation_gold_mask
                )  # B, N
                loss = kldiv.sum() / num_annotated_samples
            elif args.mask_type == 'sigmoid':
                criterion = FocalLoss(gamma=args.focal_loss_gamma)
                # bce_criterion = nn.BCEWithLogitsLoss()
                loss = criterion(
                    pred_attn,
                    annotation_gold
                )



            logging_dict["volume_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.volume_attention_loss_lambda * loss
            
            if isinstance(side_attn, torch.Tensor):
                # attend to cancer side
                cancer_side_mask = (
                    batch["cancer_laterality"][:, :2].sum(-1) == 1
                ).float()  # only one side is positive
                cancer_side_gold = batch["cancer_laterality"][
                    :, 1
                ]  # left side (seen as lung on right) is positive class
                num_annotated_samples = max(cancer_side_mask.sum(), 1)

                pred_attn = torch.exp(
                    model_output["volume_attention_{}".format(attn_num)]
                )
                side_attn = (side_attn * pred_attn.unsqueeze(-1)).sum(1)
                side_attn_log = F.log_softmax(side_attn, dim=-1)

                loss = (
                    F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none")
                    * cancer_side_mask
                ).sum() / num_annotated_samples
                logging_dict[
                    "volume_side_attention_loss_{}".format(attn_num)
                ] = loss.detach()
                total_loss += args.volume_attention_loss_lambda * loss

    return total_loss * args.annotation_loss_lambda, logging_dict, predictions


def get_annotation_weak_loss(model_output, batch, model, args):
    total_loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()

    B, _, N, H, W, = model_output["activ"].shape   # B, 3, 38, 24, 24
    sigmoid = nn.Sigmoid()
    pred = sigmoid(model_output['image_attention_1']).mean(axis=(1,2))
    gt = batch['y']['cropped_fibrosis_density']
    criterion = nn.MSELoss()
    loss = criterion(pred, gt)
    logging_dict["weak_annotation_loss"] = loss.detach()
    predictions["probs"] = {}
    predictions["golds"] = {}
    predictions["probs"]['weak'] = pred.detach()
    predictions["golds"]['weak'] = gt
    
    return loss, logging_dict, predictions
        


def get_annotation_loss(model_output, batch, model, args):
    total_loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()

    B, _, N, H, W, = model_output["activ"].shape   # B, 3, 38, 24, 24

    if len(args.mask_name) == 1:
        batch["_image_annotations"] = batch['image_annotations'][0]
        batch_mask = batch["has_annotation"][0]
        batch["_annotation_areas"] = batch["annotation_areas"][0]
        
    for attn_num in [1, 2, 3]:
        if len(batch['image_annotations']) < attn_num:
            if model_output.get("image_attention_{}".format(attn_num), None) is not None: # in this branch, only image_attention_1 is available.
                batch["_image_annotations"] = batch['image_annotations'][0]
                batch_mask = batch["has_annotation"][0]
                batch["_annotation_areas"] = batch["annotation_areas"][0]
                
            else:
                continue
        if len(args.mask_name) == 2: # if mask is multiple, redefine batch
            batch["_image_annotations"] = batch['image_annotations'][attn_num-1]
            batch_mask = batch["has_annotation"][attn_num-1]
            batch["_annotation_areas"] = batch["annotation_areas"][attn_num-1]

        side_attn = -1
        if model_output.get("image_attention_{}".format(attn_num), None) is not None: # in this branch, only image_attention_1 is available.
                                                                                      # in Simple_AttentionPool_MultiImg
                                                                                      # image_attention -> B, T, WH (log of softmax output)
                
            if len(batch["_image_annotations"].shape) == 4:
                batch["_image_annotations"] = batch["_image_annotations"].unsqueeze(1) # convert into 5D tensor.  if channel is not exist, add it. 

            # resize annotation to 'activ' size
            annotation_gold = F.interpolate(
                batch["_image_annotations"], (N, H, W), mode=args.mask_interpolation
            )
            # annotation_gold = annotation_gold * batch_mask[:, None, None, None, None] # interpolate되었더라도, batch 단위로 rule out. why??

            # renormalize scores
            mask_area = annotation_gold.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            mask_area[mask_area == 0] = 1 # depth wise mask sum
            if args.mask_type == 'softmax':
                annotation_gold /= mask_area
            elif args.mask_type == 'sigmoid':
                pass

            # reshape annotation into 1D vector
            annotation_gold = annotation_gold.view(B, N, -1).float()

            # get mask over annotation boxes in order to weigh
            # non-annotated scores with zero when computing loss
            annotation_gold_mask = (annotation_gold > 0).float() # B, N, WH

            num_annotated_samples = (annotation_gold.view(B * N, -1).sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples) # number of annotated slices

            if args.mask_type == 'softmax':
                pred_attn = (
                    model_output["image_attention_{}".format(attn_num)]  #B, T, WH  (B, 38, 576)
                    * batch_mask[:, None, None] # it is weird use batch_mask to predict attention. maybe it considering partial label??  #TODO remove it. 
                ) # TODO validate this block when mask_type is softmax
                kldiv = (
                    F.kl_div(pred_attn, annotation_gold, reduction="none")
                    * annotation_gold_mask # this multiply intend to allow that negative-annotated area is ignored. 어텐션 꼭 줘야하는것들은 loss 적용하고, no label이더라도 attention들어가는걸 막지는 않는것. 좀더 유연하다. 
                )

                # sum loss per volume and average over batches
                loss = kldiv.sum() / num_annotated_samples
            elif args.mask_type == 'sigmoid':
                pred_attn = model_output["image_attention_{}".format(attn_num)] 
                criterion = FocalLoss(gamma=args.focal_loss_gamma)
                # bce_criterion = nn.BCEWithLogitsLoss()
                loss = criterion(
                    pred_attn.view(B, N, -1),
                    annotation_gold
                )
            logging_dict["image_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.image_attention_loss_lambda * loss
           
        if model_output.get("volume_attention_{}".format(attn_num), None) is not None:
            # find size of annotation box per slice and normalize
            annotation_gold = batch["_annotation_areas"].float() * batch_mask[:, None] # TODO: area_per_slice may have been wrong from origimal repo, otherwise, it is misnaming. ckeck it. 
            if N != args.num_images:
                annotation_gold = F.interpolate(annotation_gold.unsqueeze(1), (N), mode= 'linear', align_corners = True)[:,0]
            if args.mask_type == 'softmax': 
                # Normalize area_per_slice on depth level
                area_per_slice = annotation_gold.sum(-1).unsqueeze(-1)
                area_per_slice[area_per_slice == 0] = 1
                annotation_gold /= area_per_slice
            elif args.mask_type == 'sigmoid':
                pass
            num_annotated_samples = (annotation_gold.sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples)

            # find slices with annotation
            annotation_gold_mask = (annotation_gold > 0).float()

            pred_attn = (
                model_output["volume_attention_{}".format(attn_num)]
                * batch_mask[:, None]
            )

            if args.mask_type == 'softmax':
                kldiv = (
                    F.kl_div(pred_attn, annotation_gold, reduction="none")
                    * annotation_gold_mask
                )  # B, N
                loss = kldiv.sum() / num_annotated_samples
            elif args.mask_type == 'sigmoid':
                # bce_criterion = nn.BCEWithLogitsLoss()
                criterion = FocalLoss(gamma=args.focal_loss_gamma)
                loss = criterion(
                    pred_attn,
                    annotation_gold
                )



            logging_dict["volume_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.volume_attention_loss_lambda * loss
            
        if args.dynamic_anno_lambda:
            progress = model.trainer.current_epoch / model.trainer.max_epochs
            return total_loss * args.annotation_loss_lambda * (1-progress), logging_dict, predictions
        else:
            return total_loss * args.annotation_loss_lambda, logging_dict, predictions
            

