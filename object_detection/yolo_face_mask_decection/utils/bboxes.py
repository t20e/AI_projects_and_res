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
        # x.shape => (64, 3, 448, 448) labels.shape => (64, 7, 7, 13)
        x, labels = x.to(config.DEVICE), labels.to(config.DEVICE)

        # Predicted without gradient decent
        with torch.no_grad():
            predictions = model(x)  # predictions.shape (64, 637)
        
        batch_size = x.shape[0]
        true_bboxes = output_to_bboxes(labels, config)
        pred_bboxes = output_to_bboxes(predictions, config)
        
    
    return all_pred_bboxes, all_true_bboxes


def output_to_bboxes(bboxes, config):
    """
        Re-organizes and resizes the output bboxes tensors.
        
    # TODO continure here
    """
    resize_bbox_relative_to_img(bboxes)
    # print(out.shape)
    # converted_pred = resize_bbox_relative_to_img(out).reshape(out.shape[0], config.S * config.S, -1) # reshapes the output from convert_cellboxes from (1, 7, 7, 6) to (1, 49, 6), this removes one dimension, and reorganizes the data
    # converted_pred[..., 0] = converted_pred[..., 0].long() # convert the value/tensor at the first index of the last dimension from float to int, we had values of 1.6000e+01 this just converts it into 16
    batch_size = bboxes.shape[0] #amount of images, if we had seven images output of yolo would be (7, 1470)

def resize_bbox_relative_to_img(bboxes, config):
    """
    Converts the bounding boxes from being relative to single cell that contains its midpoint to being relative to the entire image.
    
    Parameters
    ----------
        bboxes : torch.Tensor
            The bounding boxes to convert. Shape (Batch_size, NUM_NODES_PER_IMG or (num_classes + 5 * num_bboxes)) ex: (64, 637) for default config.
        config : argparse.Namespace
            Namespace object contain the configurations.
            
    Returns
    -------
    """
    bboxes = bboxes.to("cpu") # move tensor to cpu
    batch_size = bboxes.shape[0]
    bboxes = bboxes.reshape(batch_size, config.S, config.S, config.NUM_NODES_PER_CELL)
    
    bboxes1 = bboxes[..., 10:14] # grab the first bounding box (Pc1, X, Y, W, H) for every cell. shape of (1, 7, 7, 4), the coordinates of bounding boxes 1 in every cell
    # bboxes2 = predictions[..., 26:30] # grab the second box (Pc2, X, Y, W, H) for every cell. shape of (1, 7, 7, 4), the coordinates of bounding boxes 2 in every cell
    print(bboxes1)