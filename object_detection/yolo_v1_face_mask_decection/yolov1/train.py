# train the model script
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import Dataset 
from .loss import YoloLoss

from utils.checkpoints import save_checkpoint, load_checkpoint

from utils.mean_average_precision import mean_average_precision
from utils.misc import generate_model_file_name
        

def train(config, model, optimizer, transforms):
    """
    Train the model, root for training
    
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
        
    loss_fn = YoloLoss(config)    
    
    for epoch in range(config.EPOCHS):
        print("\n\n" + "|" + "-" * 64 + "|")
        print("Epoch:", epoch + 1)
        
        # Compute mean_average_percision
        mean_average_prec = mean_average_precision(loader=train_loader, model=model, config=config)
        print(f"\nTrain mAP: {mean_average_prec}")

        # Check Point optional save model if Mean Average Percision is > than num
        # if mean_average_prec > 0.9:
        #     checkpoint = {
        #         "state_dict" : model.state_dict(),
        #         "optimizer" : optimizer.state_dict()
        #     }
        #     save_checkpoint(checkpoint, filename=config.LOAD_MODEL_FILE)
    
        train_model(train_loader, model, optimizer, loss_fn, config)
    
    # Save checkpoint after training the model for a certain number of epochs
    checkpoint = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    save_checkpoint(checkpoint, filename=generate_model_file_name("Yolov1", "facemask", "objectDetection", config.EPOCHS))
    
    
# <------------- Train Pipeline ------------->
    
def train_model(train_loader, model, optimizer, loss_fn, config):
    """Run gradient descent to train the model"""
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # TODO implemenet a plot predictions while training the model, but do it at random epochs somewhere closer to the end of training.
        if config.PLOT_WHILE_TRAINING:
            pass
        #update the progress bar
        loop.set_postfix(loss=loss.item())
        
    print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")