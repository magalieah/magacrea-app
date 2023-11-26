import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import PhotoDatasetAugmented
from autoencoder import AE


IM_SIZE = 256
    
if __name__== '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
    'batch_size':4,
    'lr':1e-3,
    'epochs':100,
    'image_size':IM_SIZE,
    'train_augmentation':False,
    'test_augmentation':False,
    'num_workers':2
    }

    run = wandb.init(project='magacrea', job_type='train', save_code=True, config=config)
    config = run.config

    train_dataset = PhotoDatasetAugmented('../../data/dessin/train_dessin/bw', '../../data/photo/', train=config.train_augmentation)
    test_dataset = PhotoDatasetAugmented('../../data/dessin/test_dessin/bw',  '../../data/photo/', train=config.test_augmentation) 
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    autoencoder = AE().to(device)
    #print(summary(autoencoder, (3,256,256), batch_size=config.batch_size, device=device))
    # Validation using MSE Loss function
    loss_function = nn.MSELoss()
    # Using an Adam Optimizer with lr = 0.001
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = config.lr)

    for epoch in trange(config.epochs):
        for ix, batch in enumerate(iter(train_dataloader)):
            photo, drawing = batch
            photo, drawing = photo.to(device), drawing.to(device)

            # Shape [BATCH_SIZE, CHANNEL, SIZE, SIZE]
            # Output of Autoencoder

            reconstructed_drawing = autoencoder(photo)

            # Calculating the loss function
            loss = loss_function(reconstructed_drawing, drawing)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Storing the losses in a list for plotting
            run.log({'training/loss' : loss.item()})
            if ix==0:
                run.log({'training/reconstructed': wandb.Image(reconstructed_drawing), 'training/drawing': wandb.Image(drawing), 'training/photo': wandb.Image(photo)})


    with torch.no_grad():
        losses=[]
        autoencoder=autoencoder.eval()
        for ix, batch in enumerate(iter(test_dataloader)):
            photo, drawing = batch
            photo, drawing = photo.to(device), drawing.to(device)
            reconstructed_drawing = autoencoder(photo)
            loss = loss_function(reconstructed_drawing, drawing)
            losses.append(loss)
            run.log({'test/reconstructed': wandb.Image(reconstructed_drawing), 'test/drawing': wandb.Image(drawing), 'test/photo': wandb.Image(photo)})

        mean_loss = torch.Tensor(losses).mean()
        run.log({'test/mean_loss': mean_loss})
        autoencoder=autoencoder.train()

    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    artifact = wandb.Artifact(name="autoencoder", type="model")
    artifact.add_file(local_path="./autoencoder.pth")  # Add dataset directory to artifact
    run.log_artifact(artifact)  # Logs the artifact version "my_data:v0"
    run.finish()
