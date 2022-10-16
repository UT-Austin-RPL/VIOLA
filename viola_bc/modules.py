import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# >>> fn = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)

import robomimic
from robomimic.models.base_nets import CropRandomizer, RNN_Base, Randomizer
import torch.distributions as D

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

USE_GPU = torch.cuda.is_available()
DEVICE = TorchUtils.get_torch_device(try_to_use_cuda=True)
def safe_cuda(x):
    if USE_GPU:
        return x.cuda()
    return x

def get_activate_fn(activation):
    if activation == 'relu':
        activate_fn = torch.nn.ReLU
    elif activation == 'leaky-relu':
        activate_fn = torch.nn.LeakyReLU
    return activate_fn

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h
        
    def forward(self, x):
        assert(x.shape[1] == self._in_c)
        assert(x.shape[2] == self._in_h)
        assert(x.shape[3] == self._in_w)

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints

class ResnetConv(torch.nn.Module):
    def __init__(self,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3,
                 last_c=None,
                 no_stride=False):

        super().__init__()

        assert(remove_layer_num <= 5)
        # For training policy
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-remove_layer_num]
        if img_c != 3:
            # If use eye_in_hand, we need to increase the channel size
            conv0 = torch.nn.Conv2d(img_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1
        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        self.remove_layer_num = remove_layer_num

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        
        if self.remove_layer_num == 2:
            out_c = 512
            scale = 32.
        elif self.remove_layer_num == 3:
            out_c = 256
            scale = 16.
        elif self.remove_layer_num == 4:
            out_c = 128
            scale = 8.
        elif self.remove_layer_num == 5:
            out_c = 64
            scale = 4.

        if self.no_stride:
            scale = scale / 4.
        out_h = int(math.ceil(input_shape[1] / scale))
        out_w = int(math.ceil(input_shape[2] / scale))
        return (out_c, out_h, out_w)

class ResnetKeypoints(nn.Module):
    def __init__(self,
                 input_shape,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 num_kp=32,
                 visual_feature_dimension=64):
        super().__init__()
        self._resnet_conv = ResnetConv(pretrained=pretrained,
                                       no_training=no_training,
                                       activation=activation,
                                       remove_layer_num=remove_layer_num,
                                       img_c=input_shape[0])

        self._resnet_output_shape = self._resnet_conv.output_shape(input_shape)
        self._spatial_softmax = SpatialSoftmax(in_c=self._resnet_output_shape[0],
                                               in_h=self._resnet_output_shape[1],
                                               in_w=self._resnet_output_shape[2],
                                               num_kp=num_kp)
        self._visual_feature_dimension = visual_feature_dimension
        self._fc = torch.nn.Sequential(torch.nn.Linear(num_kp * 2, visual_feature_dimension))

    def forward(self, x):
        out = self._resnet_conv(x)
        out = self._spatial_softmax(out)
        out = self._fc(out)
        return out

    def output_shape(self, input_shape):
        return (self._resnet_output_shape[0], self._visual_feature_dimension)
        
    
# Simple components for building transformer model
    
class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head_output=64, dropout=0.):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))


    def forward(self, x):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)

        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # q.dot(k.transpose)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        self.att_weights = self.attention_fn(qk_dot_product)

        # (..., num_heads, seq_len, dim_head_output)
        out = torch.matmul(self.att_weights, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Merge the output from heads to get single vector
        return self.output_layer(out)

class SelfAttentionMasked(nn.Module):
    def __init__(self, dim, num_heads=2, dim_head_output=64, dropout=0.):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)

        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # q.dot(k.transpose)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        qk_dot_product = qk_dot_product.masked_fill(mask==1., -torch.inf)
        self.att_weights = self.attention_fn(qk_dot_product)
        # (..., num_heads, seq_len, dim_head_output)
        out = torch.matmul(self.att_weights, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Merge the output from heads to get single vector
        return self.output_layer(out)

class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Remember the residual connection
        self.layers = [nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers,
                 num_heads,
                 dim_head_output,
                 mlp_dim,
                 dropout,
                 # position_embedding_type,
                 **kwargs):
        super().__init__()

        # self.position_embedding_fn = get_position_embedding(position_embedding_type, **kwargs)
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Norm(input_dim),
                SelfAttention(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout),
                Norm(input_dim),
                TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))

            self.attention_output[_] = None

    def forward(self, x):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            x = x + drop_path(att(att_norm(x)))
            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers,
                 num_heads,
                 dim_head_output,
                 mlp_dim,
                 dropout,
                 T=1,
                 # position_embedding_type,
                 **kwargs):
        super().__init__()

        # self.position_embedding_fn = get_position_embedding(position_embedding_type, **kwargs)
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Norm(input_dim),
                SelfAttentionMasked(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout),
                Norm(input_dim),
                TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))

            self.attention_output[_] = None

        self.seq_len = None
        self.num_elements = None

    def compute_mask(self, input_shape):
        if self.num_elements is None or self.seq_len is None or self.num_elements != input_shape[2] or self.seq_len != input_shape[1]:
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (torch.triu(torch.ones(self.seq_len, self.seq_len)) - torch.eye(self.seq_len, self.seq_len)).to(DEVICE)
            self.mask = self.original_mask.repeat_interleave(self.num_elements, dim=-1).repeat_interleave(self.num_elements, dim=-2)
        
    def forward(self, x, mask=None):

        assert(mask is not None or self.mask is not None)
        if mask is not None:
            self.mask = mask
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            x = x + drop_path(att(att_norm(x), self.mask))
            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x
    

# ROI Alignment
class RoIAlignWrapper(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True,
                 bbox_size=4):
        super().__init__()
        assert(aligned==True)
        self.output_size = output_size
        self.bbox_size = bbox_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned=aligned)

    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_size = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = out.reshape(batch_size, bbox_size, channel_size, *self.output_size)
        return out

    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (input_shape[0], self.output_size[0], self.output_size[1])

class RoIAlignLayerNormWrapper(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True,
                 bbox_size=4):
        super().__init__()
        assert(aligned==True)
        self.output_size = output_size
        self.bbox_size = bbox_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned=aligned)

        self.layer_norm = nn.LayerNorm([input_shape[0], output_size[0], output_size[1]])

    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_size = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = out.reshape(batch_size, bbox_size, channel_size, *self.output_size)
        out = self.layer_norm(out)
        return out

    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (input_shape[0], self.output_size[0], self.output_size[1])
    
    
class RoIAlignSpatialSoftmaxWrapper(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True,
                 bbox_size=4,
                 num_kp=32):
        super().__init__()
        assert(aligned==True)
        self.output_size = output_size
        self.bbox_size = bbox_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned=aligned)
        self.spatial_softmax = SpatialSoftmax(in_c=input_shape[0], in_h=output_size[0], in_w=output_size[1], num_kp=num_kp)
        self.num_kp = num_kp

    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_size = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = self.spatial_softmax(out)
        out = out.reshape(batch_size, bbox_size, self.num_kp * 2)
        # out = out.reshape(batch_size, bbox_size, channel_size, *self.output_size)
        return out

    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (self.bbox_size, self.num_kp * 2)
    
    
# Image augmentation
    
class IdentityAug(nn.Module):
    def __init__(self,
                 input_shape=None,
                 *args,
                 **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def output_shape(self, input_shape):
        return input_shape

class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """
    def __init__(
            self,
            input_shape,
            translation,
    ):
        super().__init__()
        self.pad = nn.ReplicationPad2d(translation)
        pad_output_shape = (input_shape[0], input_shape[1] + translation, input_shape[2] + translation)
        self.crop_randomizer = CropRandomizer(input_shape=pad_output_shape,
                                              crop_height=input_shape[1],
                                              crop_width=input_shape[2])

    def forward(self, x):
        if self.training:
            out = self.pad(x)
            out = self.crop_randomizer.forward_in(out)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape

class TranslationAugGroup(nn.Module):
    """
    Add translation augmentation to a group of images, applying the sam e translation)
    """
    def __init__(
            self,
            input_shapes,
            translation,
    ):
        super().__init__()

        self.pad = nn.ReplicationPad2d(translation)

        self.channels = []
        for input_shape in input_shapes:
            self.channels.append(input_shape[0])
        pad_output_shape = (sum(self.channels), input_shape[1] + translation, input_shape[2] + translation)
        self.crop_randomizer = CropRandomizer(input_shape=pad_output_shape,
                                              crop_height=input_shape[1],
                                              crop_width=input_shape[2])

    def forward(self, x_groups):
        if self.training:
            x = torch.cat(x_groups, dim=1)
            out = self.pad(x)
            out = self.crop_randomizer.forward_in(out)
            out = torch.split(out, self.channels, dim=1)
            return out
        else:
            out = x_groups
        return out

class ShiftBBoxTranslationAugGroup(nn.Module):
    """
    Add translation augmentation to a group of images, applying the sam e translation)
    """
    def __init__(
            self,
            input_shapes,
            translation,
    ):
        super().__init__()

        self.pad_translation = translation//2

        self.pad = nn.ReplicationPad2d(self.pad_translation)

        self.channels = []
        for input_shape in input_shapes:
            self.channels.append(input_shape[0])
        pad_output_shape = (sum(self.channels), input_shape[1] + translation, input_shape[2] + translation)
        self.crop_randomizer = CustomCropRandomizer(input_shape=pad_output_shape,
                                              crop_height=input_shape[1],
                                              crop_width=input_shape[2])

    def forward(self, x_groups):
        if self.training:
            x = torch.cat(x_groups, dim=1)
            out = self.pad(x)
            out, crop_inds = self.crop_randomizer.forward_in(out)
            out = torch.split(out, self.channels, dim=1)
            shift_pixels = torch.cat((crop_inds[..., 1:2], crop_inds[..., 0:1], crop_inds[..., 1:2], crop_inds[..., 0:1]), dim=-1) - self.pad_translation
            return out, shift_pixels
        else:
            out = x_groups
        return out    

class ShiftBBoxTemporalTranslationAugGroup(nn.Module):
    """
    Add translation augmentation to a group of images, applying the sam e translation)
    """
    def __init__(
            self,
            input_shapes,
            translation,
    ):
        super().__init__()

        self.pad_translation = translation//2

        self.pad = nn.ReplicationPad2d(self.pad_translation)

        self.channels = []
        for input_shape in input_shapes:
            self.channels.append(input_shape[0])
        pad_output_shape = (sum(self.channels), input_shape[1] + translation, input_shape[2] + translation)
        self.crop_randomizer = TemporalCropRandomizer(input_shape=pad_output_shape,
                                              crop_height=input_shape[1],
                                              crop_width=input_shape[2])

    def forward(self, x_groups):
        if self.training:
            x = torch.cat(x_groups, dim=-3) # [..., C1 + C2, H, W]
            out = TensorUtils.time_distributed(x, self.pad)
            # out = self.pad(x)
            out, crop_inds = self.crop_randomizer.forward_in(out)
            out = torch.split(out, self.channels, dim=2)
            shift_pixels = torch.cat((crop_inds[..., 1:2], crop_inds[..., 0:1], crop_inds[..., 1:2], crop_inds[..., 0:1]), dim=-1) - self.pad_translation
            return out, shift_pixels
        else:
            out = x_groups
        return out

class CustomCropRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height, 
        crop_width, 
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CustomCropRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        out, crop_inds = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height, 
            crop_width=self.crop_width, 
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return TensorUtils.join_dimensions(out, 0, 1), crop_inds

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0, 
            target_dims=(batch_size, self.num_crops))
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg

class TemporalCropRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height, 
        crop_width, 
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(TemporalCropRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        out, crop_inds = self.sample_temporal_random_image_crops(
            images=inputs,
            crop_height=self.crop_height, 
            crop_width=self.crop_width, 
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        # return TensorUtils.join_dimensions(out, 0, 1), crop_inds
        return out, crop_inds

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0, 
            target_dims=(batch_size, self.num_crops))
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg


    def sample_temporal_random_image_crops(self, images, crop_height, crop_width, num_crops, pos_enc=False):
        """
        For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
        @images.
        Args:
            images (torch.Tensor): batch of images of shape [..., T, C, H, W]
            crop_height (int): height of crop to take

            crop_width (int): width of crop to take
            num_crops (n): number of crops to sample
            pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial 
                encoding of the original source pixel locations. This means that the
                output crops will contain information about where in the source image 
                it was sampled from.
        Returns:
            crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width) 
                if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)
            crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
        """
        device = images.device

        # maybe add 2 channels of spatial encoding to the source image
        source_im = images
        # make sure sample boundaries ensure crops are fully within the images
        T, image_c, image_h, image_w = source_im.shape[-4:]
        max_sample_h = image_h - crop_height
        max_sample_w = image_w - crop_width

        # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
        # Each gets @num_crops samples - typically this will just be the batch dimension (B), so 
        # we will sample [B, N] indices, but this supports having more than one leading dimension,
        # or possibly no leading dimension.
        #
        # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
        crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-4], num_crops).to(device)).long().expand(-1, source_im.shape[-4])
        crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-4], num_crops).to(device)).long().expand(-1, source_im.shape[-4])
        crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1) # shape [..., N, 2]

        crops = ObsUtils.crop_image_from_indices(
            images=source_im, 
            crop_indices=crop_inds, 
            crop_height=crop_height, 
            crop_width=crop_width, 
        )

        return crops, crop_inds
    

# Processing bounding box coordinates

class BBoxIdentity(nn.Module):
    """
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def forward(self, x):
        return x

class BBoxNorm(nn.Module):
    """
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            *args,
            **kwargs
    ):
        super().__init__()
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        return x

class BBoxNormFactor(nn.Module):
    """
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            *args,
            **kwargs
    ):
        super().__init__()
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var

        factor = nn.Parameter(torch.ones(1))
        self.register_parameter("factor", factor)
        

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        return x * self.factor

class Rel3DPosAug(nn.Module):
    """
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            use_noise=False,
            position_var=0.01,
            *args,
            **kwargs
    ):
        super().__init__()
        self.use_noise = use_noise
        self.position_var = position_var

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            x += torch.randn_like(x) * self.position_var
            # noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            # x += noise
        # x = torch.divide(x, self.scaling_ratio)
        return x

class Rel2DPosAug(nn.Module):
    """
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            scaling_ratio=128.,            
            use_noise=False,
            position_var=4,
            *args,
            **kwargs
    ):
        super().__init__()
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.position_var = position_var

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            x += torch.randn_like(x) * self.position_var
        x = torch.divide(x, self.scaling_ratio)
        return x
    
class BBoxCenterNorm(nn.Module):
    """
    Add normalization while also transforming from corner to center
    Args:
      scaling_ratio (float): Scaling ratio to scale bbox coordinates to [0., 1.]
      use_noise (bool): use noise for bbox coordinate value or not
      pixel_var (int): integeer of number of pixels shifted during noise.
    """
    def __init__(
            self,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
    ):
        super().__init__()
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x = torch.cat([(x[..., 0:1] + x[..., 2:3]) / 2,
                           (x[..., 1:2] + x[..., 3:]) / 2], dim=-1)
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        return x

    
# Some position embedding based on bounding box coordinates
# This should follow after bbox_aug

class BBoxPositionEmbedding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()
        self.bbox_dim = 4

    def forward(self, x, normalized_bbox):
        return torch.cat((x, normalized_bbox), axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.bbox_dim, )

class Rel3DPositionEmbedding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()
        self.dim = 9

    def forward(self, x, rel_pos):
        return torch.cat((x, rel_pos), axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.dim, )


class Rel2DPositionEmbedding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()
        self.dim = 6

    def forward(self, x, rel_pos):
        return torch.cat((x, rel_pos), axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.dim, )
    
class BBoxSinusoidalPositionEmbeddingNorm(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            dim=4,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / dim) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var
        
    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        self.position_embedding_tensor = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        self.position_embedding_tensor = torch.flatten(self.position_embedding_tensor, start_dim=-2)
        return  self.position_embedding_tensor

    def output_shape(self, input_shape):
        return input_shape

class BBoxSinusoidalPositionEmbeddingNormFactor(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            dim=4,
            factor_ratio=10.,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / dim) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var

        factor = nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)
        
    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        self.position_embedding_tensor = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        self.position_embedding_tensor = torch.flatten(self.position_embedding_tensor, start_dim=-2)
        return  self.position_embedding_tensor * self.factor

    def output_shape(self, input_shape):
        return input_shape

class BBoxTrueSinusoidalPositionEncoding(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,            
            dim=4,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var

        
    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        
        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        self.spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        return self.spatial_pos_embedding

class BBoxTrueSinusoidalPositionEncodingFactor(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,            
            dim=4,
            factor_ratio=1.,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels

        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var
        factor = nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        
        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        self.spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        return self.spatial_pos_embedding * self.factor
    
    
class BBoxSinZeroNorm(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            dim=4,
            factor_ratio=0.,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / dim) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var
        
    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            # x += torch.randn_like(x) * self.pixel_var
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        self.position_embedding_tensor = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        self.position_embedding_tensor = torch.flatten(self.position_embedding_tensor, start_dim=-2)
        return  self.position_embedding_tensor * 0.

    def output_shape(self, input_shape):
        return input_shape
    
class BBoxAdditivePositionEmbedding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def forward(self, x, normalized_bbox):
        return x + normalized_bbox

    def output_shape(self, input_shape):
        return input_shape

class BBoxCatPositionEmbedding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()
        self.bbox_dim = 4

    def forward(self, x, normalized_bbox):
        return torch.cat((x, normalized_bbox), axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.bbox_dim, )

class BBoxCatSinusoidalPositionEmbedding(nn.Module):
    def __init__(
            self,
            input_shape,            
            *args,
            **kwargs
    ):
        super().__init__()
        self.bbox_dim = input_shape[1]

    def forward(self, x, normalized_bbox):
        return torch.cat((x, normalized_bbox), axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.bbox_dim, )    

class BBoxCenterPositionEmbedding(BBoxPositionEmbedding):
    def __init__(self):
        super().__init__()

        self.bbox_dim = 2

class TemporalSinusoidalPositionEncoding(nn.Module):
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)
        
    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        # print(pos_x.shape)
        # print(self.inv_freq.shape)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        # emb = torch.zeros((x.shape[1], self.channels), device=x.device).type(x.type())
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

class TemporalZeroPositionEncoding(nn.Module):
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)
        
    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        # print(pos_x.shape)
        # print(self.inv_freq.shape)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * 0.

    def output_shape(self, input_shape):
        return input_shape
    
class PatchBBoxSinsuisoidalPositionEncoding(torch.nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            dim=4,
            factor_ratio=1.,
            num_proposals=20,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        
        x_steps = int(np.sqrt(num_proposals)) + 1
        y_steps = num_proposals // (x_steps - 1) + 1

        px = torch.linspace(0, 128, steps=x_steps)
        py = torch.linspace(0, 128, steps=y_steps)
        bbox_list = []
        for j in range(len(py) - 1):
            for i in range(len(px) - 1):
                bbox_list.append(torch.tensor([px[i], py[j], px[i+1], py[j+1]]))
        
        bbox_tensor = torch.stack(bbox_list, dim=0)
        print(bbox_tensor.shape)
        x = torch.divide(bbox_tensor, scaling_ratio)
        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        self.register_buffer("bbox", bbox_tensor)
        self.register_buffer("spatial_pos_embedding", spatial_pos_embedding)
        self.scaling_ratio = scaling_ratio
        factor = torch.nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)

    def forward(self):
        return self.spatial_pos_embedding * self.factor
    
    def get_bbox_list(self, batch_size):
        return [self.bbox] * batch_size
    
class FlattenProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert(len(input_shape) == 4), "You should not use FlattenProjection if not having a tensor with 4 dimensions (excluding batch dimension)"
        in_dim = input_shape[-3] * input_shape[-2] * input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=2)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)

class SpatialProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert(len(input_shape) == 3), "You should not use FlattenProjection if not having a tensor with 3 dimensions (excluding batch dimension)"

        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)

class SpatialLinearProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert(len(input_shape) == 3), "You should not use FlattenProjection if not having a tensor with 3 dimensions (excluding batch dimension)"

        in_dim = input_shape[-3] * input_shape[-2] * input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)
    
class LinearProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim,
                 bias=True):
        super().__init__()

        in_dim = input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        out = self.projection(x)
        return out

    def output_shape(self, input_shape):
        if len(input_shape) > 1:
            return input_shape[:1] + (self.out_dim,)
        else:
            return (self.out_dim, )

class ProprioProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim,
                 bias=True,
                 squash=False):
        super().__init__()

        in_dim = input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim, bias=bias)
        if squash:
            self.squash_layer = nn.Tanh()
        else:
            self.squash_layer = nn.Identity()

    def forward(self, x):
        out = self.projection(x)
        out = self.squash_layer(out)
        return out

class CatProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim,
                 bias=True,
                 squash=False):
        super().__init__()

        in_dim = input_shape[-1]
        self.out_dim = out_dim
        zeros = torch.zeros((out_dim - in_dim))
        self.register_buffer("zeros", zeros)
        # self.projection = nn.Linear(in_dim, out_dim, bias=bias)
        # if squash:
        #     self.squash_layer = nn.Tanh()
        # else:
        #     self.squash_layer = nn.Identity()

    def forward(self, x):
        batch_size = x.shape[0]
        out = torch.cat([x, self.zeros.unsqueeze(0).expand(batch_size, -1)], dim=-1)
        return out
    

class MLPLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 num_dim=1024,
                 activation='relu'):
        super().__init__()
        self.output_dim = output_dim
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                activate_fn()]

            self._layers += [torch.nn.Linear(num_dim, output_dim)]
        else:
            self._layers += [torch.nn.Linear(input_dim, output_dim)]
        self.layers = torch.nn.Sequential(*self._layers)

    def forward(self, x):
        h = self.layers(x)
        return h
    
    
class PolicyMLPLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 num_dim=1024,
                 activation='relu',
                 action_scale=1.,
                 action_squash=True):
        super().__init__()
        self.output_dim = output_dim
        self.action_scale = 1.0
        self.action_squash = action_squash
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                activate_fn()]

            self._layers += [torch.nn.Linear(num_dim, output_dim)]
        else:
            self._layers += [torch.nn.Linear(input_dim, output_dim)]
        self.layers = torch.nn.Sequential(*self._layers)

    def forward(self, x):
        h = self.layers(x)
        if self.action_squash:
            h = torch.tanh(h) * self.action_scale
        return h

class GMMPolicyMLPLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 num_layers=2,
                 num_dim=1024,              
                 mlp_activation="relu",
                 std_activation="softplus", 
                 low_noise_eval=True, 
                 use_tanh=False):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        if mlp_activation == 'relu':
            mlp_activate_fn = torch.nn.ReLU
        elif mlp_activation == 'leaky-relu':
            mlp_activate_fn = torch.nn.LeakyReLU
        
        out_dim = self.num_modes * self.output_dim
        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            mlp_activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                mlp_activate_fn()]

        else:
            self._layers += [torch.nn.Linear(input_dim, num_dim)]
        self.mlp_layers = torch.nn.Sequential(*self._layers)
        
        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        self.mean_layer = nn.Linear(num_dim, out_dim)
        self.scale_layer = nn.Linear(num_dim, out_dim)
        self.logits_layer = nn.Linear(num_dim, self.num_modes)
        
    def forward(self, x):
        x = self.mlp_layers(x)
        means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(x)
        
        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.scales = scales
        self.logits = logits

        return dist


class TemporalGMMPolicyMLPLayer(nn.Module):
    """This is a mlp layer that handles temporal sequence. (because of of restricted usage from robomimic)
    """
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 num_layers=2,
                 num_dim=1024,              
                 mlp_activation="relu",
                 std_activation="softplus", 
                 low_noise_eval=True, 
                 use_tanh=False):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        if mlp_activation == 'relu':
            mlp_activate_fn = torch.nn.ReLU
        elif mlp_activation == 'leaky-relu':
            mlp_activate_fn = torch.nn.LeakyReLU
        
        out_dim = self.num_modes * self.output_dim
        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            mlp_activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                mlp_activate_fn()]

        else:
            self._layers += [torch.nn.Linear(input_dim, num_dim)]
        self.mlp_layers = torch.nn.Sequential(*self._layers)
        
        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        self.mean_layer = nn.Linear(num_dim, out_dim)
        self.scale_layer = nn.Linear(num_dim, out_dim)
        self.logits_layer = nn.Linear(num_dim, self.num_modes)

    def forward_fn(self, x):
        out = self.mlp_layers(x)
        means = self.mean_layer(out).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(out).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(out)

        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std
        
        return means, scales, logits
        
    def forward(self, x):
        # x = self.mlp_layers(x)
        # means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        # scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        # logits = self.logits_layer(x)

        means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.scales = scales
        self.logits = logits

        return dist    
    
class EyeInHandKeypointNet(nn.Module):
    def __init__(self,
                 # shallow_channels=64,
                 img_h=128,
                 img_w=128,
                 num_kp=16,
                 visual_feature_dimension=32,
                 use_translation_aug=False,
                 no_grad=False,
                 remove_layer_num=4):
        super().__init__()
        # self.encoder = eval(encoder.network)(**encoder.network_kwargs)
        # self.shallow_perception_net = ShallowPerceptionEmbedding(output_channels=shallow_channels)

        self.use_translation_aug = use_translation_aug
        self.no_grad = no_grad
        if self.use_translation_aug:
            self.aug = TranslationAug(input_shape=(3, img_w, img_h), translation=4)
        
        self.encoder = ResnetConv(remove_layer_num=remove_layer_num, img_c=3)
        # Temporarily hard coded
        encoder_output_shape = self.encoder.output_shape(input_shape=(3, img_h, img_w))
        self.spatial_softmax = SpatialSoftmax(in_c=encoder_output_shape[0], in_h=encoder_output_shape[1], in_w=encoder_output_shape[2], num_kp=num_kp)
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_kp * 2, visual_feature_dimension))
        self.visual_feature_dimension = visual_feature_dimension
        
    def forward(self, x):
        if not self.no_grad:
            if self.use_translation_aug:
                x = self.aug(x)
            x = self.encoder(x)
            x = self.spatial_softmax(x)
            x = self.fc(x)
        else:
            with torch.no_grad():
                if self.use_translation_aug:
                    x = self.aug(x)
                x = self.encoder(x)
                x = self.spatial_softmax(x)
                x = self.fc(x)            
        return x


class BCCatGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        if self.use_eye_in_hand:
            self.eye_in_hand_encoder = EyeInHandKeypointNet()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

    def forward(self, input_tensor, obs_dict):
        if self.num_modalities == 0:
            return input_tensor

        tensor_list = [input_tensor]
        if self.use_eye_in_hand:
            tensor_list.append(self.eye_in_hand_encoder(obs_dict["eye_in_hand_rgb"]))
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_gripper_history:
            tensor_list.append(obs_dict["gripper_history"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])

        # for tensor in tensor_list:
        #     print(tensor.shape)

        
        return torch.cat(tensor_list, dim=-1)
        
    def output_shape(self, input_shape, shape_meta):
        dim = input_shape[-1]
        if self.use_eye_in_hand:
            dim += self.eye_in_hand_encoder.visual_feature_dimension
        if self.use_joint:
            dim += shape_meta["all_shapes"]["joint_states"][0]
        if self.use_gripper:
            dim += shape_meta["all_shapes"]["gripper_states"][0]
        if self.use_gripper_history:
            dim += shape_meta["all_shapes"]["gripper_history"][0]
        if self.use_ee:
            dim += shape_meta["all_shapes"]["ee_states"][0]
        return (dim,)
    
    
class CatGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        if "eye_in_hand" not in kwargs:
            kwargs["eye_in_hand"] = {}
        if self.use_eye_in_hand:
            self.eye_in_hand_encoder = EyeInHandKeypointNet(**kwargs["eye_in_hand"])
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

    def forward(self, input_tensor, obs_dict):
        if self.num_modalities == 0:
            return input_tensor

        tensor_list = [input_tensor]
        if self.use_eye_in_hand:
            tensor_list.append(self.eye_in_hand_encoder(obs_dict["eye_in_hand_rgb"]))
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_gripper_history:
            tensor_list.append(obs_dict["gripper_history"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])

        # for tensor in tensor_list:
        #     print(tensor.shape)

        
        return torch.cat(tensor_list, dim=-1)
        
    def output_shape(self, input_shape, shape_meta):
        dim = input_shape[-1] * input_shape[0]
        if self.use_eye_in_hand:
            dim += self.eye_in_hand_encoder.visual_feature_dimension
        if self.use_joint:
            dim += shape_meta["all_shapes"]["joint_states"][0]
        if self.use_gripper:
            dim += shape_meta["all_shapes"]["gripper_states"][0]
        if self.use_gripper_history:
            dim += shape_meta["all_shapes"]["gripper_history"][0]
        if self.use_ee:
            dim += shape_meta["all_shapes"]["ee_states"][0]
        return (dim,)

class AveragingGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 average_all=False):
        # average_all is to mark if we should average all the
        # modalities, or average just the output from transformer
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        if self.use_eye_in_hand:
            self.eye_in_hand_encoder = EyeInHandKeypointNet()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee
        self.average_all = average_all

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

    def forward(self, input_tensor, obs_dict):
        if self.num_modalities == 0:
            return input_tensor
        tensor_list = [input_tensor]
        # if not self.average_all:
        if self.use_eye_in_hand:
            tensor_list.append(self.eye_in_hand_encoder(obs_dict["eye_in_hand_rgb"]))
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_gripper_history:
            tensor_list.append(obs_dict["gripper_history"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])
        return torch.cat(tensor_list, dim=-1)
        # else:
        #     raise NotImplementedError
        
    def output_shape(self, input_shape, shape_meta):
        dim = input_shape[-1]
        # if not self.average_all:
        if self.use_eye_in_hand:
            dim += self.eye_in_hand_encoder.visual_feature_dimension
        if self.use_joint:
            dim += shape_meta["all_shapes"]["joint_states"][0]
        if self.use_gripper:
            dim += shape_meta["all_shapes"]["gripper_states"][0]
        if self.use_gripper_history:
            dim += shape_meta["all_shapes"]["gripper_history"][0]
        if self.use_ee:
            dim += shape_meta["all_shapes"]["ee_states"][0]
        return (dim,)

    

class CatSeqGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 embedding_size=32,
                 learnable_position=False,
                 learnable_position_size=16):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        self.nets = nn.ModuleDict()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.embedding_size = embedding_size
        if self.use_eye_in_hand:
            self.nets["eye_in_hand_rgb"] = EyeInHandKeypointNet(visual_feature_dimension=embedding_size)
        if self.use_joint:
            self.nets["joint_states"] = LinearProjection(input_shape=(7,), out_dim=embedding_size)

        if self.use_gripper:
            self.nets["gripper_states"] = LinearProjection(input_shape=(2,), out_dim=embedding_size)

        if self.use_gripper_history:
            self.nets["gripper_history"] = LinearProjection(input_shape=(10, ), out_dim=embedding_size)
        if self.use_ee:
            self.nets["ee_states"] = LinearProjection(input_shape=(3, ), out_dim=embedding_size)

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

        self.learnable_position = learnable_position
        self.learnable_position_size = learnable_position_size
        if self.learnable_position:
            self.learnable_position_dict = {"input_tensor": nn.Parameter(torch.randn(1, self.learnable_position_size)).to(DEVICE)}
            for obs_key in self.nets.keys():
                self.learnable_position_dict[obs_key] = nn.Parameter(torch.randn(1, self.learnable_position_size)).to(DEVICE)
        
    def forward(self, input_tensor, obs_dict):
        if self.num_modalities == 0:
            return input_tensor
        tensor_list = [input_tensor.unsqueeze(1)]
        batch_size = input_tensor.shape[0]
        for obs_key, net in self.nets.items():
            out = net(obs_dict[obs_key])
            if self.learnable_position:
                out = torch.cat([out,
                                 self.learnable_position_dict[obs_key].repeat(batch_size, 1)
                ], dim=-1)
            tensor_list.append(out.unsqueeze(1))
        return torch.cat(tensor_list, dim=1)

    def output_shape(self, input_shape, shape_meta):
        dim = input_shape[0]
        for obs_key in self.nets.keys():
            dim += 1

        if self.learnable_position:
            dim += self.learnable_position
        return (dim, input_shape[-1])


class ActionTokenGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 embedding_size=32,
                 joint_states_dim=7,
                 gripper_states_dim=2,
                 gripper_history_dim=10,                 
                 squash=False,
                 zero_action_token=True,                 
                 *args,
                 **kwargs):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        self.nets = nn.ModuleDict()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.embedding_size = embedding_size

        if zero_action_token:
            action_token = nn.Parameter(torch.zeros(embedding_size))
        else:
            action_token = nn.Parameter(torch.randn(embedding_size))
        
        self.register_parameter("action_token", action_token)
        if "eye_in_hand" not in kwargs:
            kwargs["eye_in_hand"] = {}        
        if self.use_eye_in_hand:
            self.nets["eye_in_hand_rgb"] = EyeInHandKeypointNet(visual_feature_dimension=embedding_size, **kwargs["eye_in_hand"])
        if self.use_joint:
            self.nets["joint_states"] = ProprioProjection(input_shape=(joint_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper:
            self.nets["gripper_states"] = ProprioProjection(input_shape=(gripper_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper_history:
            self.nets["gripper_history"] = ProprioProjection(input_shape=(gripper_history_dim, ), out_dim=embedding_size, squash=squash)
        if self.use_ee:
            self.nets["ee_states"] = ProprioProjection(input_shape=(3, ), out_dim=embedding_size, squash=squash)

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

        
    def forward(self, bbox_input_tensor, obs_dict):
        batch_size = bbox_input_tensor.shape[0]

        self.tensor_list = [self.action_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1),
                            bbox_input_tensor]
        if self.num_modalities == 0:
            return torch.cat(self.tensor_list, dim=1)

        batch_size = bbox_input_tensor.shape[0]
        for obs_key, net in self.nets.items():
            out = net(obs_dict[obs_key])
            self.tensor_list.append(out.unsqueeze(1))
        return torch.cat(self.tensor_list, dim=1)

    def output_shape(self, input_shape, shape_meta):
        dim = 2
        for obs_key in self.nets.keys():
            dim += 1
        return (dim, input_shape[-1])
    

class ActionTokenSpatialContextGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 embedding_size=32,
                 joint_states_dim=7,
                 gripper_states_dim=2,
                 gripper_history_dim=10,
                 squash=False,
                 zero_action_token=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        self.nets = nn.ModuleDict()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.embedding_size = embedding_size

        if zero_action_token:
            action_token = nn.Parameter(torch.zeros(embedding_size))
        else:
            action_token = nn.Parameter(torch.randn(embedding_size))
        self.register_parameter("action_token", action_token)
        if "eye_in_hand" not in kwargs:
            kwargs["eye_in_hand"] = {}        
        if self.use_eye_in_hand:
            self.nets["eye_in_hand_rgb"] = EyeInHandKeypointNet(visual_feature_dimension=embedding_size, **kwargs["eye_in_hand"])
        if self.use_joint:
            self.nets["joint_states"] = ProprioProjection(input_shape=(joint_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper:
            self.nets["gripper_states"] = ProprioProjection(input_shape=(gripper_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper_history:
            self.nets["gripper_history"] = ProprioProjection(input_shape=(gripper_history_dim, ), out_dim=embedding_size, squash=squash)
        if self.use_ee:
            self.nets["ee_states"] = ProprioProjection(input_shape=(3, ), out_dim=embedding_size, squash=squash)

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

        
    def forward(self, spatial_context_input_tensor, bbox_input_tensor, obs_dict):
        batch_size = bbox_input_tensor.shape[0]
        self.tensor_list = [self.action_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1),
                            spatial_context_input_tensor.unsqueeze(1),
                            bbox_input_tensor]
        if self.num_modalities == 0:
            return torch.cat(self.tensor_list, dim=1)

        batch_size = bbox_input_tensor.shape[0]
        for obs_key, net in self.nets.items():
            out = net(obs_dict[obs_key])
            self.tensor_list.append(out.unsqueeze(1))
        return torch.cat(self.tensor_list, dim=1)

    def output_shape(self, input_shape, shape_meta):
        dim = 3
        for obs_key in self.nets.keys():
            dim += 1
        return (dim, input_shape[-1])
 
class ActionTokenSpatialContextNoObjectsGroupModalities(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=False,
                 use_joint=False,
                 use_gripper=False,
                 use_gripper_history=False,
                 use_ee=False,
                 embedding_size=32,
                 joint_states_dim=7,
                 gripper_states_dim=2,
                 gripper_history_dim=10,
                 squash=False,
                 zero_action_token=True,                 
                 *args,
                 **kwargs):
        super().__init__()
        self.use_eye_in_hand = use_eye_in_hand
        self.nets = nn.ModuleDict()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_gripper_history = use_gripper_history
        self.use_ee = use_ee

        self.embedding_size = embedding_size

        if zero_action_token:
            action_token = nn.Parameter(torch.zeros(embedding_size))
        else:
            action_token = nn.Parameter(torch.randn(embedding_size))
        self.register_parameter("action_token", action_token)
        if "eye_in_hand" not in kwargs:
            kwargs["eye_in_hand"] = {}        
        if self.use_eye_in_hand:
            self.nets["eye_in_hand_rgb"] = EyeInHandKeypointNet(visual_feature_dimension=embedding_size, **kwargs["eye_in_hand"])
        if self.use_joint:
            self.nets["joint_states"] = ProprioProjection(input_shape=(joint_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper:
            self.nets["gripper_states"] = ProprioProjection(input_shape=(gripper_states_dim,), out_dim=embedding_size, squash=squash)

        if self.use_gripper_history:
            self.nets["gripper_history"] = ProprioProjection(input_shape=(gripper_history_dim, ), out_dim=embedding_size, squash=squash)
        if self.use_ee:
            self.nets["ee_states"] = ProprioProjection(input_shape=(3, ), out_dim=embedding_size, squash=squash)

        self.num_modalities = int(self.use_eye_in_hand) + int(self.use_joint) + int(self.use_gripper) + int(self.use_ee) + int(self.use_gripper_history)

        
    def forward(self, spatial_context_input_tensor, obs_dict):
        batch_size = spatial_context_input_tensor.shape[0]
        self.tensor_list = [self.action_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1),
                            spatial_context_input_tensor.unsqueeze(1)]
        if self.num_modalities == 0:
            return torch.cat(self.tensor_list, dim=1)

        batch_size = spatial_context_input_tensor.shape[0]
        for obs_key, net in self.nets.items():
            out = net(obs_dict[obs_key])
            self.tensor_list.append(out.unsqueeze(1))
        return torch.cat(self.tensor_list, dim=1)

    def output_shape(self, input_shape, shape_meta):
        dim = 2
        for obs_key in self.nets.keys():
            dim += 1
        return (dim, input_shape[-1])

class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """
    def __init__(
            self,
            input_shape,
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.3,
            epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(brightness=brightness, 
                                                               contrast=contrast, 
                                                               saturation=saturation, 
                                                               hue=hue)
        self.epsilon = epsilon

    def forward(self, x):
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape

class DataAugGroup(torch.nn.Module):
    """
    Add augmentation to multiple inputs
    """
    def __init__(
            self,
            use_color_jitter=True,
            use_random_erasing=False,
            **aug_kwargs
    ):
        super().__init__()

        transforms = []

        self.use_color_jitter = use_color_jitter
        self.use_random_erasing = use_random_erasing

        if self.use_color_jitter:
            color_jitter = torchvision.transforms.ColorJitter(**aug_kwargs["color_jitter"])
            transforms.append(color_jitter)
        if self.use_random_erasing:
            random_erasing = torchvision.transforms.RandomErasing(**aug_kwargs["random_erasing"])
            transforms.append(random_erasing)

        self.transforms = torchvision.transforms.Compose(transforms)

    def forward(self, x_groups):
        split_channels = []
        for i in range(len(x_groups)):
            split_channels.append(x_groups[i].shape[0])
        if self.training:
            x = torch.cat(x_groups, dim=0)
            out = self.transforms(x)
            out = torch.split(out, split_channels, dim=0)
            return out
        else:
            out = x_groups
        return out

class DataAugStackedImgGroup(torch.nn.Module):
    """
    Add augmentation to stacked images
    """
    def __init__(
            self,
            use_color_jitter=True,
            use_random_erasing=False,
            **aug_kwargs
    ):
        super().__init__()

        transforms = []

        self.use_color_jitter = use_color_jitter
        self.use_random_erasing = use_random_erasing

        if self.use_color_jitter:
            color_jitter = torchvision.transforms.ColorJitter(**aug_kwargs["color_jitter"])
            transforms.append(color_jitter)
        if self.use_random_erasing:
            random_erasing = torchvision.transforms.RandomErasing(**aug_kwargs["random_erasing"])
            transforms.append(random_erasing)

        self.transforms = torchvision.transforms.Compose(transforms)

    def forward(self, x):
        if self.training:
            x = rearrange(x, 'b (n c) h w -> (b n) c h w', c=3)
            out = self.transforms(x)
            out = rearrange(out, '(b n) c h w -> b (n c) h w', n=2)
            return out
        else:
            out = x
        return out

class RNNBackbone(nn.Module):
    def __init__(self,
                 input_dim=64,
                 rnn_hidden_dim=1000,
                 rnn_num_layers=2,
                 rnn_type="LSTM",
                 per_step_net=None,
                 rnn_kwargs={"bidirectional": False},
                 *args,
                 **kwargs):
        super().__init__()
        self.per_step_net = eval(per_step_net.network)(**per_step_net.network_kwargs)
        self.rnn_model = Robomimic_RNN_Base(
            input_dim=64,
            rnn_hidden_dim=1000,
            rnn_num_layers=2,
            rnn_type="LSTM",
            per_step_net=self.per_step_net,
            rnn_kwargs=rnn_kwargs
        )

    def forward(self, x, *args, **kwargs):
        return self.rnn_model(x, *args, **kwargs)

    def get_rnn_init_state(self, *args, **kwargs):
        return self.rnn_model.get_rnn_init_state(*args, **kwargs)

    def forward_step(self, *args, **kwargs):
        return self.rnn_model.forward_step(*args, **kwargs)

class GMMPolicyOutputHead(robomimic.models.base_nets.Module):
    """GMM policy output head without any nonlinear MLP layers."""
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 std_activation="softplus", 
                 low_noise_eval=False, 
                 use_tanh=False
    ):
        super().__init__()

        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std

        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        out_dim = self.num_modes * output_dim

        self.mean_layer = nn.Linear(input_dim, out_dim)
        self.scale_layer = nn.Linear(input_dim, out_dim)
        self.logits_layer = nn.Linear(input_dim, self.num_modes)
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        
    def forward(self, x):
        means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(x)
        
        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        self.means = means
        self.scales = scales
        self.logits = logits

        return {"means": self.means,
                "scales": self.scales,
                "logits": self.logits}


class RPNPretrained(torch.nn.Module):
    def __init__(self,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3,
                 last_c=None,
                 no_stride=False):

        super().__init__()
        from detection_bc.centernet_module import load_centernet_rpn

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.model = load_centernet_rpn().model.backbone

    def forward(self, x):
        with torch.no_grad():
            h = self.model(x)['p3']
        return h

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        out_c = 256
        scale = 8.

        if self.no_stride:
            scale = scale / 4.
        out_h = int(math.ceil(input_shape[1] / scale))
        out_w = int(math.ceil(input_shape[2] / scale))
        return (out_c, out_h, out_w)