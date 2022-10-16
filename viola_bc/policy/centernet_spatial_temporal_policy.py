from torch import nn
from viola_bc.modules import *
from viola_bc.decoders import *
from viola_bc.policy.base_policy import *
import torchvision

import robomimic.utils.tensor_utils as TensorUtils


class CenterNetSpatialTemporalPolicy(BasePolicy):
    def __init__(self,
                 policy_cfg,
                 shape_meta):
        super().__init__()

        self.policy_cfg = policy_cfg
        input_shape = shape_meta["all_shapes"]["agentview_rgb"]
        obs_keys = list(shape_meta["all_shapes"].keys())

        self.bbox_name = None
        for key in obs_keys:
            if "centernet" in key or "bbox" in key:
                self.bbox_name = key
                break
        if self.bbox_name is None:
            raise ValueError
        original_img_scale = input_shape[-1]

        #####################################################
        ###
        ### Augmentation
        ###
        #####################################################
        self.data_aug = eval(policy_cfg.data_aug.network)(**policy_cfg.data_aug.network_kwargs)

        policy_cfg.img_aug.network_kwargs["input_shapes"] = (shape_meta["all_shapes"]["agentview_rgb"], shape_meta["all_shapes"]["eye_in_hand_rgb"])        
        self.img_aug = eval(policy_cfg.img_aug.network)(**policy_cfg.img_aug.network_kwargs)


        #####################################################
        ###
        ### Encoder
        ###
        #####################################################
        self.encoder = eval(policy_cfg.encoder.network)(**policy_cfg.encoder.network_kwargs)
        input_shape = self.encoder.output_shape(input_shape)
        # You need compute spatial scale for roi align

        #####################################################
        ###
        ### Spatial Projection ( same as the following projection)
        ###
        #####################################################        
        policy_cfg.spatial_projection.network_kwargs["input_shape"] = input_shape
        policy_cfg.spatial_projection.network_kwargs["out_dim"] = policy_cfg.projection.network_kwargs["out_dim"]
        print(input_shape)
        self.spatial_projection = eval(policy_cfg.spatial_projection.network)(**policy_cfg.spatial_projection.network_kwargs)
        
        #####################################################
        ###
        ### Object-centric Pooling
        ###
        #####################################################
        if "spatial_scale" in policy_cfg.pooling.network_kwargs:
            spatial_scale = input_shape[2] / original_img_scale
            policy_cfg.pooling.network_kwargs["spatial_scale"] = spatial_scale
        policy_cfg.pooling.network_kwargs["input_shape"] = input_shape
        self.pooling = eval(policy_cfg.pooling.network)(**policy_cfg.pooling.network_kwargs)
        input_shape = self.pooling.output_shape(input_shape)
        print("after pooling: ", input_shape)

        #####################################################
        ###
        ### Projection
        ###
        #####################################################
        input_shape = (shape_meta["all_shapes"][self.bbox_name][0], ) + input_shape        
        policy_cfg.projection.network_kwargs["input_shape"] = input_shape
        print(input_shape)
        self.projection = eval(policy_cfg.projection.network)(**policy_cfg.projection.network_kwargs)
        input_shape = self.projection.output_shape(input_shape)

        #####################################################
        ###
        ### Processing bbox coordinates, usually with added noise
        ###
        #####################################################
        policy_cfg.bbox_norm.network_kwargs["input_shape"] = input_shape
        self.bbox_norm = eval(policy_cfg.bbox_norm.network)(**policy_cfg.bbox_norm.network_kwargs)

        #####################################################
        ###
        ### BBox position embedding
        ###
        #####################################################
        self.bbox_position_embedding = eval(policy_cfg.bbox_position.network)(**policy_cfg.bbox_position.network_kwargs)
        input_shape = self.bbox_position_embedding.output_shape(input_shape)

        # Look at how previous decoder is implemented
        self.grouping = eval(policy_cfg.grouping.network)(**policy_cfg.grouping.network_kwargs)
        input_shape = self.grouping.output_shape(input_shape, shape_meta)

        policy_cfg.temporal_position.network_kwargs.input_shape = input_shape
        self.temporal_position_encoding = eval(policy_cfg.temporal_position.network)(**policy_cfg.temporal_position.network_kwargs)
        input_shape = self.temporal_position_encoding.output_shape(input_shape)

        # Initialize transformer
        input_dim = input_shape[-1]
        self.transformer = eval(policy_cfg.transformer.network)(input_dim=input_dim,
                                                                **policy_cfg.transformer.network_kwargs)
        
        #####################################################
        ###
        ### Decoder (Policy output head)
        ###
        #####################################################
        # If we use MLP only, we will concatenate tensor in the policy model
        policy_cfg.decoder.network_kwargs.input_dim = input_shape[-1]
        policy_cfg.decoder.network_kwargs.output_dim = shape_meta["ac_dim"]
        self.decoder = eval(policy_cfg.decoder.network)(**policy_cfg.decoder.network_kwargs)

        self.max_len = 10

    def encode_fn(self, data):
        #####################################################
        ### augmentation on images
        #####################################################        
        batch_size = data["obs"]["agentview_rgb"].shape[0]
        out = self.img_aug((data["obs"]["agentview_rgb"], data["obs"]["eye_in_hand_rgb"]))
        out, data["obs"]["eye_in_hand_rgb"] = self.data_aug(out)        
        

        #####################################################
        ### Encoder
        #####################################################                
        self.encoder_out = self.encoder(out)

        # Add a spatial softma layer to get spatial_projection_out
        self.spatial_projection_out = self.spatial_projection(self.encoder_out)
        #####################################################
        ### Pooling on spatial feature maps
        #####################################################        
        bbox = data["obs"][self.bbox_name]
        bbox_list = bbox_batch_to_list(bbox)
        self.pooling_out = self.pooling(self.encoder_out, bbox_list)
        self.projection_out = self.projection(self.pooling_out)

        #####################################################
        ### Process bbox coordinates
        #####################################################                
        normalized_bbox = self.bbox_norm(bbox)
        #####################################################
        ### Position embedding out
        #####################################################
        self.position_embedding_out = self.bbox_position_embedding(self.projection_out, normalized_bbox)
        self.position_embedding_out = self.grouping(self.spatial_projection_out, self.position_embedding_out, data["obs"])

        return self.position_embedding_out

    def decode_fn(self, x, per_step=False):
        # print(x.shape, self.temporal_position_encoding)
        self.temporal_positions = self.temporal_position_encoding(x)
        self.temporal_out = x + self.temporal_positions.unsqueeze(0).unsqueeze(2)
        original_shape = self.temporal_out.shape
        self.transformer.compute_mask(self.temporal_out.shape)
        flattened_temporal_out = TensorUtils.join_dimensions(self.temporal_out, 1, 2)
        transformer_out = self.transformer(flattened_temporal_out)
        transformer_out = transformer_out.reshape(original_shape)
        action_token_out = transformer_out[:, :, 0, :]
        if per_step:
            action_token_out = action_token_out[:, -1:, :]
        action_outputs = self.decoder(action_token_out)
        return action_outputs

    def forward(self, data):
        data = self.process_input_for_training(data)
        out = TensorUtils.time_distributed(data, self.encode_fn)
        batch_size, seq_len = out.shape[:2]
        dist = self.decode_fn(out)
        return dist
    
    def get_action(self, data):
        data = TensorUtils.to_device(data, self.device)
        data = self.process_input_for_evaluation(data)
        with torch.no_grad():
            # encode_out = self.encode_fn(data)
            encode_out = TensorUtils.time_distributed(data, self.encode_fn)
            self.queue.append(encode_out)
            if len(self.queue) > self.max_len:
                self.queue.pop(0)
            temporal_sequence = torch.cat(self.queue, dim=1)
            dist = self.decode_fn(temporal_sequence, per_step=True)
        return dist.sample().detach().cpu().squeeze().numpy()
        
    def reset(self):
        self.queue = []

    def process_input_for_training(self, x):
        return x

    def process_input_for_evaluation(self, x):
        return TensorUtils.recursive_dict_list_tuple_apply(
            x,
            {
                torch.Tensor: lambda x: x.unsqueeze(dim=0).unsqueeze(dim=0),
            }
        )
    @property
    def device(self):
        return next(self.parameters()).device
