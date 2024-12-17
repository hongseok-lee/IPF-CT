import torch
import torch.nn as nn

class MultiAttentionPool(nn.Module):
    def __init__(self):
        super(MultiAttentionPool, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
class Attention3DPool(nn.Module):
    def __init__(self, reduction_ratio=8, **kwargs):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        super(Attention3DPool, self).__init__()
        self.reduction_ratio = reduction_ratio

        # 3D convolution for spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(kwargs['num_chan'], kwargs['num_chan'] // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(kwargs['num_chan'] // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel attention (squeeze and excitation)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(kwargs['num_chan'], kwargs['num_chan'] // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(kwargs['num_chan'] // reduction_ratio, kwargs['num_chan'], kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.size()

        # Spatial attention
        spatial_attn = self.spatial_attn(x)  # (B, 1, T, H, W)
        
        # Channel attention
        channel_attn = self.channel_attn(x)  # (B, C, 1, 1, 1)

        # Apply both attentions
        attended = x * spatial_attn * channel_attn

        # Global average pooling
        output = torch.mean(attended, dim=(2, 3, 4))  # (B, C)

        return {
            'hidden': output, # (B,C)
            'spatial_attention': spatial_attn.squeeze(1),  # (B, T, H, W)
            'channel_attention': channel_attn.squeeze(-1).squeeze(-1).squeeze(-1),  # (B, C)
            'attended': attended  # (B, C, T, H, W)
        }
    
    
    
    

class MultiAttentionPoolSig(nn.Module):
    def __init__(self):
        super(MultiAttentionPoolSig, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg_Sig(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
class MultiAttentionPool_2D_GAP(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_2D_GAP, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = AttentionPool2D(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = AttentionPool2D(**params) # MaxPool on HW
        self.volume_pool2 = Simple_AttentionPool(**params)
        # self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        # self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
    
    
class MultiAttentionPool_2D(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_2D, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = AttentionPool2D(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
    
    
class MultiAttentionPoolDual(nn.Module):
    def __init__(self):
        super(MultiAttentionPoolDual, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool2 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
class MultiAttentionPoolDualNoGMP(nn.Module):
    def __init__(self):
        super(MultiAttentionPoolDualNoGMP, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool2 = Simple_AttentionPool(**params) # Depth-wise 1d-attention
        
        self.image_pool3 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool3 = Conv1d_AttnPool(**params)

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        image_pool_out3 = self.image_pool3(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out3 = self.volume_pool3(image_pool_out3['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
            
        for key, val in image_pool_out3.items():
            output[f'{key}_3'] = val
        for key, val in volume_pool_out3.items():
            output[f'{key}_3'] = val
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 ) #[B, 1024, 38]
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()
                                                                                            #[B, 38, 1024] -> [B, 38, 512]     [B, 512, 38]
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], volume_pool_out3['hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
    
class GlobalMaxPool(nn.Module):
    '''
    Pool to obtain the maximum value for each channel
    '''
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. output['hidden'] is (B, C)
        '''
        spatially_flat_size = (*x.size()[:2], -1) # B, C, TWH
        x = x.view(spatially_flat_size)
        hidden, _ = torch.max(x, dim=-1) # B, C
        return {'hidden': hidden}

class PerFrameMaxPool(nn.Module):
    '''
    Pool to obtain the maximum value for each slice in 3D input 
    '''
    def __init__(self):
        super(PerFrameMaxPool, self).__init__()
    
    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. 
                + output['multi_image_hidden'] is (B, C, T)
        '''
        assert len(x.shape) == 5
        B, C, T, W, H = x.size()
        output = {}
        spatially_flat_size = (B, C, T, W*H) # (B, C, T, WH)
        x = x.view(spatially_flat_size) # (B, C, T, WH)
        output['multi_image_hidden'], _ = torch.max(x, dim=-1)  # (B, C, T)
        return output

class Conv1d_AttnPool(nn.Module):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        '''
        Pool to learn an attention over the slices after convolution
        '''
        def __init__(self, **kwargs):
            super(Conv1d_AttnPool, self).__init__()
            self.conv1d = nn.Conv1d(kwargs['num_chan'], kwargs['num_chan'], kernel_size= kwargs['conv_pool_kernel_size'], stride=kwargs['stride'], padding= kwargs['conv_pool_kernel_size']//2, bias=False)
            self.aggregate = Simple_AttentionPool(**kwargs) # B, C, T -> volume_attention: (B,T) / hidden: (B,C)
        
        def forward(self, x):
            '''
            args: 
                - x: tensor of shape (B, C, T)
            returns:
                - output: dict
                    + output['attention_scores']: tensor (B, C)
                    + output['hidden']: tensor (B, C)
            '''
            # X: B, C, N
            x = self.conv1d(x) # B, C, N'
            return self.aggregate(x)

class Simple_AttentionPool(nn.Module):
    # input image_pool_out1['multi_image_hidden']
        # multi_image_hidden -> B, C, T
        
    '''
    Pool to learn an attention over the slices
    '''
    def __init__(self, **kwargs):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        super(Simple_AttentionPool, self).__init__()

        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        '''
        output = {}
        B, C, _ = x.shape
        spatially_flat_size = (B, C, -1) #B, C, -1 #meanless

        x = x.view(spatially_flat_size) #B, C, -1 -> B, C, N #meanless
        attention_scores = self.attention_fc(x.transpose(1,2)) #B, N, C ->  B, N, 1 -> slice attention map
                                                        
        output['volume_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1) # logsoftmax(B, 1, N) -> (B, N) slice attention map
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #softmax(B, 1, N, dim=-1)   # Depth-wise attention
        output['volume_attention_vis'] = attention_scores
        
        x = x * attention_scores #B, C, N
        output['hidden'] = torch.sum(x, dim=-1) # (B,C)
        return output

    
class Simple_AttentionPool_MultiImg(nn.Module):

    '''
    Pool to learn an attention over the slices and the volume
    '''
    def __init__(self, **kwargs):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        super(Simple_AttentionPool_MultiImg, self).__init__()

        self.attention_fc = nn.Linear(kwargs['num_chan'], 1) # 512,1
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
            1, 512, 38  , 24, 24
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, T, C)
                + output['multi_image_hidden']: tensor (B, T, C)
                + output['hidden']: tensor (B, T*C)
        '''
        output = {} 
        B, C, T, W, H = x.size() # 1, 512, 38  , 24, 24
        x = x.permute([0,2,1,3,4]) # B T C W H  B, 38, 512, 24, 24
        x = x.contiguous().view(B*T, C, W*H) # B * 38, 512, 24*24
        attention_scores = self.attention_fc(x.transpose(1,2)) #BT, WH , 1  |   BT, WH, C -(Linear)-> BT, WH, 1
        # This implementation has limitation which can not reflectimplattention, but only 2d attentionation wh
                                                               
        output['image_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, T, -1) # BT, 1, WH   => B, T, WH  (B, 38, 576)
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #BT, 1, WH
        output['image_attention_vis'] = attention_scores
        
        x = x * attention_scores #x=BT, C, WH | attention_map (BT, 1, WH) is broadcastedsted. 
        x = torch.sum(x, dim=-1) # BT, C
        output['multi_image_hidden'] = x.view(B, T, C).permute([0,2,1]).contiguous() # B, C, T
        output['hidden'] = x.view(B, T * C)
        
        # in summaryummary, 
        # image_attention -> B, T, WH
        # multi_image_hidden -> B, C, T
        # hidden -> B, T * C
        return output
    

class Simple_AttentionPool_MultiImg_Sig(nn.Module):

    '''
    Pool to learn an attention over the slices and the volume
    '''
    def __init__(self, **kwargs):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        super(Simple_AttentionPool_MultiImg_Sig, self).__init__()

        self.attention_fc = nn.Linear(kwargs['num_chan'], 1) # 512,1
        self.activation = nn.Sigmoid()

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
            1, 512, 38  , 24, 24
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, T, C)
                + output['multi_image_hidden']: tensor (B, T, C)
                + output['hidden']: tensor (B, T*C)
        '''
        output = {} 
        B, C, T, W, H = x.size() # 1, 512, 38  , 24, 24
        x = x.permute([0,2,1,3,4]) # B T C W H  B, 38, 512, 24, 24
        x = x.contiguous().view(B*T, C, W*H) # B * 38, 512, 24*24
        attention_scores = self.attention_fc(x.transpose(1,2)) #BT, WH , 1  |   BT, WH, C -(Linear)-> BT, WH, 1
        # This implementation has limitation which can not reflect 3D attention, but only 2d attention(BT, WH)
                                                               
        output['image_attention'] = attention_scores.transpose(1,2).view(B, T, -1) # BT, 1, WH   => B, T, WH  (B, 38, 576)
        attention_scores = self.activation( attention_scores.transpose(1,2)) #BT, 1, WH
        
        x = x * attention_scores #x=BT, C, WH | attention_map (BT, 1, WH) is broadcasted. 
        x = torch.sum(x, dim=-1) # BT, C
        output['multi_image_hidden'] = x.view(B, T, C).permute([0,2,1]).contiguous() # B, C, T
        output['hidden'] = x.view(B, T * C)
        
        # in summary, 
        # image_attention -> B, T, WH
        # multi_image_hidden -> B, C, T
        # hidden -> B, T * C
        return output
    
    
class AttentionPool2D(nn.Module):

    '''
    Pool to learn an attention over the slices and the volume
    '''
    def __init__(self, **kwargs):
        # params = {
        #     'num_chan': 512,
        #     'conv_pool_kernel_size': 11,
        #     'stride': 1
        #     }
        super(AttentionPool2D, self).__init__()

        self.down1 = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.down2 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
            ) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d (256, 256, kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d (256, 256, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv2d(128, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
            1, 512, 38  , 24, 24
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, T, C)
                + output['multi_image_hidden']: tensor (B, T, C)
                + output['hidden']: tensor (B, T*C)
        '''
        output = {} 
        B, C, T, W, H = x.size() # 1, 512, 38  , 24, 24
        x = x.permute([0,2,1,3,4]) # B T C W H  B, 38, 512, 24, 24
        x = x.contiguous().view(B*T, C, W, H) # B * 38, 512, 24, 24
        
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        c1 = torch.cat([u1, d2], dim=1)
        c1 = self.conv1(c1)
        
        u2 = self.up2(c1)
        c2 = torch.cat([u2, d1], dim=1)
        c2 = self.conv2(c2)
        
        final = self.final(c2)
        attention_scores = self.sigmoid(final) #BT, 1, W, H   |   BT, C, W, H -(Linear)-> BT, 1, W, H
                                                               
        output['image_attention'] = final.view(B,T,-1) # BT, 1, W, H -> B,T,WH
        # x:  B * 38, 512, 24, 24
        x = x * attention_scores #x=B*T, C, W, H | attention_map (B*T, 1, W, H) is broadcasted. 
        x = self.gap(x).view(B*T, C) # B*T, C, 1, 1
        output['multi_image_hidden'] = x.view(B, T, C).permute([0,2,1]).contiguous() # B, C, T
        output['hidden'] = x.view(B, T * C)
        
        # in summary, 
        # image_attention -> B, T, WH
        # multi_image_hidden -> B, C, T
        # hidden -> B, T * C
        return output




class MultiAttentionPool_12(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_12, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.hidden_fc = nn.Linear(2 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
        
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 
    
    

class MultiAttentionPool_12_2D(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_12_2D, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = AttentionPool2D(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.hidden_fc = nn.Linear(2 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
        
        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_23(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_23, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.hidden_fc = nn.Linear(2 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        hidden = torch.cat( [ volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_31(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_31, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.hidden_fc = nn.Linear(2 * 512, 512)

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        hidden = torch.cat( [ volume_pool_out1['hidden'], output['maxpool_hidden']], dim = -1 ) #[ B, 512*3 ]
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_1(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_1, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.hidden_fc = nn.Linear(1 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        hidden =  volume_pool_out1['hidden']
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_1_2D(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_1_2D, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = AttentionPool2D(**params) # HW dimension 1d-attention
        self.volume_pool1 = Simple_AttentionPool(**params) # Depth-wise 1d-attention

        self.hidden_fc = nn.Linear(1 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden"[B, 512, 38], "image_attention"[B, 38, 196], "hidden"(B, 19456])
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]

        for key, val in image_pool_out1.items():
            output[f'{key}_1'] = val
        for key, val in volume_pool_out1.items():
            output[f'{key}_1'] = val
            
        hidden =  volume_pool_out1['hidden']
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_2(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_2, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }

        self.image_pool2 = PerFrameMaxPool() # MaxPool on HW
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.hidden_fc = nn.Linear(1 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"[B, 512, 38]
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden"[B, 512], "volume_attention"[B, 38]
        
        for key, val in image_pool_out2.items():
            output[f'{key}_2'] = val
        for key, val in volume_pool_out2.items():
            output[f'{key}_2'] = val
    
        hidden = volume_pool_out2['hidden']
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 

class MultiAttentionPool_3(nn.Module):
    def __init__(self):
        super(MultiAttentionPool_3, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.global_max_pool = GlobalMaxPool() # MaxPool on DHW

        self.hidden_fc = nn.Linear(1 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        #       1, 512, 38 , 24, 24
        output = {}
        
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden'] # [B, 512]
        
        hidden = output['maxpool_hidden']
        output['hidden'] = self.hidden_fc(hidden) # [B, 512*3] -> [B, 512]

        return output 