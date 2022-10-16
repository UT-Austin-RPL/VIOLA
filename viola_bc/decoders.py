from torch import nn
from viola_bc.modules import *



class CenterNetSimpleTransformer(nn.Module):
    def __init__(self,
                 shape_meta,
                 input_shape,
                 transformer_encoder,
                 policy_output_head,
                 grouping):
        super().__init__()
        input_dim = input_shape[-1]
        seq_length = input_shape[0]
        self.transformer_encoder = eval(transformer_encoder.network)(input_dim=input_dim,
                                                                     **transformer_encoder.network_kwargs)
        self.grouping = eval(grouping.network)(**grouping.network_kwargs)
        policy_input_dim  = self.grouping.output_shape(input_shape, shape_meta)
        self.policy_output_head = eval(policy_output_head.network)(input_dim=policy_input_dim[0],
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self, x, obs_dict):
        batch_size = x.shape[0]
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)

        out = self.grouping(out, obs_dict)
        out = self.policy_output_head(out)
        return out


class CenterNetSimpleTransformerV2(nn.Module):
    def __init__(self,
                 shape_meta,
                 input_shape,
                 transformer_encoder,
                 policy_output_head,
                 grouping):
        super().__init__()
        self.grouping = eval(grouping.network)(**grouping.network_kwargs)
        input_shape  = self.grouping.output_shape(input_shape, shape_meta)
        input_dim = input_shape[-1]
        # seq_length = input_shape[0]
        self.transformer_encoder = eval(transformer_encoder.network)(input_dim=input_dim,
                                                                     **transformer_encoder.network_kwargs)
        self.policy_output_head = eval(policy_output_head.network)(input_dim=input_dim,
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self, x, obs_dict):
        out = self.grouping(x, obs_dict)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        out = self.policy_output_head(out)
        return out

class GTBBoxCatDecoder(nn.Module):
    """
    In this version, we do not use transformer and directly concat bbox information
    """
    def __init__(self,
                 shape_meta,
                 input_shape,
                 policy_output_head,
                 grouping):
        super().__init__()
        self.grouping = eval(grouping.network)(**grouping.network_kwargs)
        input_shape  = self.grouping.output_shape(input_shape, shape_meta)
        input_dim = input_shape[-1]
        seq_length = input_shape[0]
        # self.transformer_encoder = eval(transformer_encoder.network)(input_dim=input_dim,
        #                                                              **transformer_encoder.network_kwargs)
        self.policy_output_head = eval(policy_output_head.network)(input_dim=input_dim * seq_length,
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self,
                spatial_context_out,
                bbox_out,                
                obs_dict):
        out = self.grouping(spatial_context_out, bbox_out, obs_dict)
        out = torch.flatten(out, start_dim=-2)
        out = self.policy_output_head(out)
        return out

    
    
class CenterNetTransformerSpatialContext(CenterNetSimpleTransformerV2):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                spatial_context_out,
                bbox_out,
                obs_dict):
        out = self.grouping(spatial_context_out, bbox_out, obs_dict)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        out = self.policy_output_head(out)
        return out

class CenterNetTransformerActionToken(CenterNetSimpleTransformerV2):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                bbox_out,
                obs_dict):
        out = self.grouping(bbox_out, obs_dict)
        out = self.transformer_encoder(out)
        out = out[:, 0, ...]

        out = self.policy_output_head(out)
        return out
    
    
class CenterNetTransformerSpatialContextActionToken(CenterNetSimpleTransformerV2):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                spatial_context_out,
                bbox_out,
                obs_dict):
        out = self.grouping(spatial_context_out, bbox_out, obs_dict)
        out = self.transformer_encoder(out)
        out = out[:, 0, ...]

        out = self.policy_output_head(out)
        return out
    
class BCDecoder(nn.Module):
    def __init__(self,
                 shape_meta,
                 input_shape,
                 policy_output_head,
                 grouping):
        super().__init__()
        input_dim = input_shape[-1]

        self.grouping = eval(grouping.network)(**grouping.network_kwargs)
        policy_input_dim  = self.grouping.output_shape(input_shape, shape_meta)
        policy_output_head.network_kwargs.output_dim = shape_meta["ac_dim"]
        self.policy_output_head = eval(policy_output_head.network)(input_dim=policy_input_dim[0],
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self, x, obs_dict):
        out = self.grouping(x, obs_dict)
        out = self.policy_output_head(out)
        return out

class SpatialContextGroupTransformerEncoder(nn.Module):
    def __init__(self,
                 shape_meta,
                 input_shape,
                 transformer_encoder,
                 grouping):
        super().__init__()
        self.grouping = eval(grouping.network)(**grouping.network_kwargs)
        input_shape  = self.grouping.output_shape(input_shape, shape_meta)
        self.input_dim = input_shape[-1]
        # seq_length = input_shape[0]
        self.transformer_encoder = eval(transformer_encoder.network)(input_dim=self.input_dim,
                                                                     **transformer_encoder.network_kwargs)

    def forward(self,
                spatial_context_out,
                bbox_out,
                obs_dict):
        out = self.grouping(spatial_context_out, bbox_out, obs_dict)
        out = self.transformer_encoder(out)
        out = out[:, 0, ...]
        return out

    def output_shape(self,
                     input_shape):
        return (input_shape[:-1], self.input_dim)
