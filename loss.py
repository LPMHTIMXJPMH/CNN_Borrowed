import torch
import torch.nn as nn

import methods

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.split = 7
        self.bbox_dense = 2
        self.class_num = 20

        self.lambda_none = 0.5
        self.lambda_coord = 5


    def coord(self, target, pred):
        # shape of target(s) : [batch size] [self.width_split_size] [self.height_split_size] [class + ]
        target_coords = target[..., 21:25]

        pred = pred.reshape(-1, self.split, self.split, self.class_num + (1 + 4) * self.bbox_dense)
        
        # pred_confidence_first_s = pred[..., 20]
        # pred_confidence_second_s = pred[..., 25]

        pred_coord_first_s = pred[..., 21:25]
        pred_coord_second_s = pred[..., 26:30]

        ious_first = methods.ious(pred_coord_first_s, target_coords)[:, 0]
        ious_second = methods.ious(pred_coord_second_s, target_coords)[:, 0]

        iouss = torch.cat([ious_first.unsqueeze(1), ious_second.unsqueeze(1)], dim = 1)
        # iouss = [ [ , ] , [ , ] , [ , ] , ... , [ , ] ]

        _, higher_iou = torch.max(iouss, dim = 1)

        pred_coord_first_second_s = torch.cat([pred_coord_first_s.unsqueeze(1), pred_coord_second_s.unsqueeze(1)], dim = 1)
        
        pred_coord = pred_coord_first_second_s[:, higher_iou]

        return target_coords, pred_coord


# torch.nn.forward() toward loss function !
# None non-maximum-supression has been used in loss function!
def forward(self, target, pred):

    # corrdinates loss
    target_somes = target[..., 20]
    target_coord, pred_coord = self.coord(target, pred)
    target_coord[2:4] = torch.sqrt(target_coord)
    pred_coord = torch.sqrt(torch.sign(pred_coord[2:4]) * pred_coord[2:4])

    coord_loss = self.mse(target_coord, pred_coord)
    
    
    
    
    
    
import torch
import torch.nn as nn

import cvConfig
# length of target is 7 * 7 * 5
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()


    def forward(self, pred, target):
        # -1 means I dont care about batch size (first dimension of prediction is batch size number, -1 mean selet all of them)
        pred = pred.reshape(-1, cvConfig.loss["split"], cvConfig.loss["split"], cvConfig.loss["guesser"]//(4+1)*5 + cvConfig.loss["num_cls"] // cvConfig.loss["qutity"])

        
