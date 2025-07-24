import torch
from .nms import non_max_suppression


def get_true_and_pred_bboxes(loader, model, config):
    """
    Get the true and predicted bounding boxes from the labels and the model output. This is for the sole-purpose of computing Mean Average Percision.
    
    Parameters
    ----------
        loader : torch.utils.data.DataLoader
            The DataLoader object containing the dataset.
        model : torch.nn.Module
            The YOLOv1 model.
        config : argparse.Namespace
            Namespace object contain the configurations.
            
    Returns
    -------
        Tuple (List[Tensor], List[Tensor])
            A tuple containing the true bounding boxes and the predicted bounding boxes.
    """
    all_pred_bboxes = []
    all_true_bboxes = []
    
    model.eval() # remove model from training mode

    train_idx = 0
    
    for batch_idx, (x, labels) in enumerate(loader):
        x, labels = x.to(config.DEVICE), labels.to(config.DEVICE) # ex shapes: x.shape => (64, 3, 448, 448) labels.shape => (64, 7, 7, 13)
        # Predicted without gradient decent
        with torch.no_grad():
            predictions = model(x)  # predictions.shape ex:(64, 637) (batch_size, NUM_PIXELS_PER_IMG)
        
        # reorder and resize the bboxes, also sending the true bboxes labels
        true_bboxes = convert_bboxes(labels, config)
        pred_bboxes = convert_bboxes(predictions, config)
        
        for idx in range(config.BATCH_SIZE):
            nms_bboxes = non_max_suppression(
                pred_bboxes[idx],
                IOU_threshold=config.IOU_THRESHOLD,
                min_threshold=config.MIN_THRESHOLD,
            )

            for pred_box in nms_bboxes:
                all_pred_bboxes.append([train_idx] + pred_box)
            
            for true_box in true_bboxes[idx]:
                if true_box[1] > config.MIN_THRESHOLD:
                    all_true_bboxes.append([train_idx] + true_box)
            
            train_idx += 1

    
    model.train()
    return all_pred_bboxes, all_true_bboxes

def convert_bboxes(bboxes, config):
    """
        Re-shapes and re-sizes bboxes tensors.
        
        Parameters
        ----------
            bboxes : tensor
                shape of (1, 637)        
        
        Returns
        -------
            Python list
                nested list of the best bounding boxes. [ [class_id, confidence_score, x1, y1, x2, y2] * for each cell in image, etc...   ]
    """
    S = config.S
    # convert the bboxes from being relative to a cells ratio to being relative to the entire image
    converted_best_bboxes = resize_bbox_relative_to_img(bboxes, config).reshape(bboxes.shape[0], S * S, -1)
    
    converted_best_bboxes[..., 0] = converted_best_bboxes[..., 0].long() # convert the value/tensor at the first index of the last dimension | i.e the confidence scores from float to int, we had values of 1.6000e+01 this just converts it into 16 class score.
    batch_size = bboxes.shape[0]
    all_bboxes = []
    
    
    # NOTE: this is just taking the tensor matrix and converting it into a python list, maybe we can just do it without convert it into a basic python list, will try for other projects
    for ex_idx in range(batch_size):
        b = []
        # loop thru each bounding box
        for bbox_idx in range(S * S):
            b.append([x.item() for x in converted_best_bboxes[ex_idx, bbox_idx, :]])

        all_bboxes.append(b)
    return all_bboxes
    
    
def resize_bbox_relative_to_img(bboxes, config):
    """
    Converts the best bounding boxes from each cell, from being relative to single cell that contains its midpoint to being relative to the entire image.
    
    Parameters
    ----------
        bboxes : torch.Tensor
            The bounding boxes to convert. Shape (Batch_size, NUM_NODES_PER_IMG or (num_classes + 5 * num_bboxes)) ex: (64, 637) for default config.
        config : argparse.Namespace
            Namespace object contain the configurations.
            
    Returns
    -------
        Tensor
            a (batch_size, 7, 7, 6) tensor containing the best bounding boxes from each cell that have been converted to the ratio of the entire image.
    
    """
    S = config.S
    bboxes = bboxes.to("cpu") # move tensor to cpu
    batch_size = bboxes.shape[0]
    bboxes = bboxes.reshape(batch_size, S, S, config.NUM_NODES_PER_CELL) #reshape from (64, 637) -> (64, 7, 7, 13)
    bboxes1 = bboxes[..., 4:config.NUM_NODES_PER_CELL-5] # grab the first bounding boxes coordinates( X, Y, W, H) for every cell. 
    bboxes2 = bboxes[..., 9:config.NUM_NODES_PER_CELL]
    
    scores = torch.cat(
        (
            bboxes[..., 3].unsqueeze(0),
            bboxes[..., 8].unsqueeze(0)
        ), dim=0
    )
    # create a tensor to store best bbox scores, ex             If the first box has a higher confidence per a cell, best_box will contain 0 at that cell vice versa for bbox2 would be a 1.
    best_box = scores.argmax(0).unsqueeze(-1)
    # now using that best_box tensor add to the bboxes1and2 to remove the the lower scored pc_1 probability score bounding boxes.
    best_bboxes = bboxes1 * ( 1 - best_box) + best_box * bboxes2
    
    # make a col like tensor to compute with the col cells and then switch it later to compute with the row cells
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    
    # Grab the x coordinate from the best bounding boxes. The 1 represents the entire image, divide by S how many cols or rows, to get thew size of a cell.
    x = 1 / S * (best_bboxes[..., :1] + cell_indices)
    # Grab the y coordinate from the best bounding boxes, also reorder the cell_indices tensor so we can multiply by the column
    y = 1 / S * (best_bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # grab teh W, H coordinates of the best bounding boxes
    w_h = 1/S * best_bboxes[..., 2:4]
    
    # concat the bounding boxes that have been converted to the ratio of entire image
    converted_best_bboxes = torch.cat((x, y, w_h), dim=-1)
    
    # grab the 3 class predictions
    predicted_classes = bboxes[..., :3].argmax(-1).unsqueeze(-1)
    
    # Grab the max/most confident from the two predicted bounding boxers for every cell
    best_confidence = torch.max(bboxes[..., 3], bboxes[..., 8]).unsqueeze(-1)
    # print(predicted_classes.shape, best_confidence.shape, converted_best_bboxes.shape)
    
    converted = torch.cat(
        # concat ex: predicted_classes = (batch_size, 7, 7, 1)  +  best_confidence = (batch_size, 7, 7, 1) +  converted_best_bboxes = (batch_size, 7, 7, 4)   ==> (batch_size, 7, 7, 6)
        
        (predicted_classes, best_confidence, converted_best_bboxes),
        dim=-1
    )    
    return converted