import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import set_random_seed
from dataset import CelebaDataset, transform, augment, inverse_normalize
from model import Discriminator, Generator


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of data loading workers')
    parser.add_argument('--input_chanels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--noise_dim', type=int, default=100, help='Noise dimention for generator input')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Coefficient for computing running averages of gradient')
    parser.add_argument('--beta2', type=float, default=0.999, help='Coefficient for computing running averages of gradient square')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--folder_path_celeba', type=str, default='../../data/celeba_hq_256', help='Path to the folder with data images')
    parser.add_argument('--folder_path_weights', type=str, default='weights', help='Path to the folder with model weights')
    parser.add_argument('--folder_path_gen_images', type=str, default='generated_images', help='Path to the folder with generated images')
    config = parser.parse_args()
    print(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    celeba_dataset = CelebaDataset(
        folder_path=config.folder_path_celeba, 
        transform=transform, 
        augment=augment
    )
    dataloader = DataLoader(
        dataset=celeba_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        drop_last=True
    )

    G = Generator(noise_dim=config.noise_dim, input_chanels=config.input_chanels).to(device)
    D = Discriminator(input_channels=config.input_chanels).to(device)

    if not os.path.exists(config.folder_path_weights):
        os.makedirs(config.folder_path_weights)
        
    if not os.path.exists(config.folder_path_gen_images):
        os.makedirs(config.folder_path_gen_images)

    criterion = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    fixed_noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)

    for epoch in range(config.num_epochs):

        G.train()
        D.train()
        
        g_loss_running = 0
        d_loss_running = 0

        # D and G play the two-player minimax game

        for batch_id, real_batch in enumerate(dataloader):
            
            # --------------------------------------------------------------------------
            # 1. Update/improve Discriminator by maximizing log(D(x)) + log(1 - D(G(z))).

            # D(x) is the probability that real image from data distribution p_data(x) 
            # is a real image.
            # D(G(z)) is the probability that the fake image from noise prior p_g(z) 
            # (the output of the generator G(z)) is a real image. 
            # --------------------------------------------------------------------------

            real_batch = real_batch.to(device)
            noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)
            fake_batch = G(noise)

            d_prob_real = D(real_batch).view(-1)
            d_prob_fake = D(fake_batch).view(-1)
            d_loss_real = criterion(d_prob_real, torch.ones_like(d_prob_real))
            d_loss_fake = criterion(d_prob_fake, torch.zeros_like(d_prob_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss_running += d_loss.item()

            D.zero_grad()
            d_loss.backward(retain_graph=True) # we will reuse fake_batch in generator below
            d_optimizer.step()

            # --------------------------------------------------------------------------
            # 2. Update/improve Generator by minimizing log(1 - D(G(z))).

            # Rather than training G to minimize log(1 − D(G(z))), we can train G to 
            # maximize log D(G(z)), because it provides much stronger gradients early 
            # in learning.
            # Generator wants to trick discriminator, so fake batch will be real for the 
            # discriminator.
            # --------------------------------------------------------------------------
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # Fake labels are real for generator cost!
            d_prob = D(fake_batch).view(-1)
            g_loss = criterion(d_prob, torch.ones_like(d_prob)) 
            g_loss_running += g_loss.item()
            
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if batch_id > 0 and batch_id % 50 == 0:
                print(
                    f"epoch={epoch}",
                    f"batch={batch_id:03d}/{len(dataloader)}",
                    # d_loss is the quality of discriminator (the lower the better)
                    f"d_loss={d_loss.item():.3f}",
                    # g_loss is the quality of generator (the lower the better) 
                    f"g_loss={g_loss.item():.3f}",
                    # D(x) is the average discriminator outputs for the real batch. 
                    # This should start close to 1 and then converge to 0.5 as G gets better.
                    f"D(x)={d_prob_real.mean().item():.3f}",  
                    # D(G(z)) is the average discriminator outputs for the fake batch. 
                    # This should start close to 0 and then converge to 0.5 as G gets better.
                    f"D(G(z))={d_prob_fake.mean().item():.3f}",
                )

        print('************************************************************************')
        print(
            f"epoch={epoch}",
            f"d_loss={d_loss_running/len(dataloader):.3f}",
            f"g_loss={g_loss_running/len(dataloader):.3f}"
        )
        print('************************************************************************')

        with torch.no_grad():
            
            fake_imgs = G(fixed_noise).detach().cpu()

            fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(7,5))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            axes = axes.flatten()
            for i in range(35):
                axes[i].imshow(inverse_normalize(fake_imgs[i]).permute(1,2,0))
                axes[i].axis('off')
            plt.savefig(f'{config.folder_path_gen_images}/img_epoch{epoch:02d}.png', dpi=300)
            plt.close()  # close the figure to avoid displaying it
            
    # save the model after the last epoch
    torch.save(G.state_dict(), f"{config.folder_path_weights}/g_model_epoch_{epoch:02d}.pt")
    torch.save(D.state_dict(), f"{config.folder_path_weights}/d_model_epoch_{epoch:02d}.pt")
    

if __name__ == "__main__":
    set_random_seed(seed=0)
    train()

# python3 train.py --batch_size 64 --num_workers 4 --input_chanels 3