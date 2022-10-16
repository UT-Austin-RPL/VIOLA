from torch import nn

def bbox_batch_to_list(bbox_tensor):
    bbox_list = [bbox for bbox in bbox_tensor]
    return bbox_list

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def process_input_for_training(self, x):
        raise NotImplementedError

    def process_input_for_evaluation(self, x):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device    
    
    def reset(self):
        pass
