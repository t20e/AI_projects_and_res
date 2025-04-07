# train the model script
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import Dataset 
from .loss import YoloLoss

from utils.bboxes import get_true_and_pred_bboxes

from utils.mean_average_precision import mean_average_precision

        
# <------------- Train Pipeline ------------->
def train(config, model, optimizer, transforms):
    """
    Train the model
    
    Parameters
    ----------
        config : argparse.Namespace
            Namescape object, contains all configurations.
        model : nn.Module
            Yolov1 model object
        optimizer : torch.optim
            optimizer object.
        transforms : torchvision.transforms
            the transform object to resize and normalize image tensors.
    """
    
    
    # def train_fn(train_loader, model, optimizer, loss_fn):
    #     loop = tqdm(train_loader, leave=True)
    #     mean_loss = []
        
    #     for batch_idx, (x, y) in enumerate(loop):
    #         x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    #         out = model(x)
    #         loss = loss_fn(out, y)
    #         mean_loss.append(loss.item())
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         #update the progress bar
    #         loop.set_postfix(loss=loss.item())
            
    #     print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")

    train_dataset = Dataset(config.DATASET_DIR, S=7, B=2, C=3, transform=transforms)
    # test_dataset = etc...

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )
    # test_loader = etc..

    if config.LOAD_MODEL:
        print("\n\nLoading Model.")
        
    for epoch in range(config.EPOCHS):
        # print("\n\n" + "|" + "-" * 64 + "|")
        # print("Epoch:", epoch + 1)
        
        # Compute mean_average_percision
        mean_average_prec = mean_average_precision(loader = train_loader, model=model, config=config)
        print(f"\nTrain mAP: {mean_average_prec}")
        # pred_bboxes, true_bboxes = get_true_and_pred_bboxes(loader=train_loader, model=model, config=config)
