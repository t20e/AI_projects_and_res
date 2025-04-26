# """Util functions to for bounding boxes"""

# from typing import Optional
# from argparse import Namespace
# from torch import Tensor, arange

# # TODO why do i need it? just keep the tensor size as 7x7x28
# def get_bboxes(all_bboxes:Tensor, config):
#     """
#     Get bounding boxes from a tensor. 

#     Note: for the models predicted bboxes we will only get the ones that pass a Non-Max-Suppression. The bboxes will be reshaped so that they're size is relative to the entire image rather than a single cell.

#     Parameters:
#         all_bboxes : torch.Tensor
#             All the bboxes for a single image, for the label it will contain zeros where they are no bboxes. For the predicted, the bboxes will need to pass a NMS, to remove weak bboxes.
#         config : Namespace
#             Configurations.
#     Returns
#     -------
#         bboxes : tensor
#             shape : (num_cells_with_bboxes, NUM_NODES_PER_CELL)
#     """
    
#     S = config.S
#     C = config.C 

#     # TODO For models predicted:
#     #   1. apply nms, and add a 1 to index 18

#     # Get the good bounding boxes
#     # create a mask, will contain true or false where a cell contains a 1 at index C.
#     # For label matrices, if a cell contains a bbox, then at index 18 will contain a 1. C=18
#     mask = all_bboxes[..., C] == 1 # C is the confidence score after the num classes
#     # use the mask to select matching cells
#     bboxes_flat = all_bboxes.view(-1, 28) # flatten ex: (7, 7, 28) -> to -> (49, 28) 
#     mask_flat = mask.view(-1)
#     # apply the mask
#     good_bboxes = bboxes_flat[mask_flat]

#     # Rescale the bounding boxes from being relative to a cell, to being relative to the entire image
#     bbox_rel_img = resize_bboxes_rel_to_img(good_bboxes, S)
    

#     return good_bboxes




# def resize_bboxes_rel_to_img(bboxes:Tensor, S:int):
#     """Resizes bboxes to be relative to the size of the entire image, rather than a single cell.
#     Parameters
#     ----------
#         bboxes : torch.Tensor
#             A tensor containing only cells with bboxes.
#         S : int
#             Split_size.
#     Returns
#     -------
#         torch.Tensor
#     """

#     # make a col like tensor to compute with the col cells and then switch it later to compute with the row cells
#     num_bboxes = bboxes.shape[0]
#     cell_indices = arange(S).repeat(num_bboxes, S, 1).unsqueeze(-1)
#     print(cell_indices.shape)
#     print(bboxes.shape)
#     # Grab the x coordinate from the bounding boxes. The 1 represents the entire image, divide by S to get the size of a single cell.
#     # x = 1 / S * (bboxes[..., :19] + cell_indices)
#     # print(x)
#     print(bboxes[..., :19])
