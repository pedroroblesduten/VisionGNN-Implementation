import torch
import torch.nn as nn
from load_data import LoadData
import os
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from models.vqvae import VQVAE
from utils import loading_bar, plot_losses


class TrainerVQVAE():
    def __init__(
        self,
        batch_size,
        device,
        epochs,
        save_model_path,
        save_losses_path,
        save_plot_path,
        save_weights_interval,
        save_reconstruction_interval,
        verbose,
        vqvae_config,
        load_data_config,
    ):
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.save_model_path = save_model_path
        self.save_losses_path = save_losses_path
        self.save_plot_path = save_plot_path
        self.save_weights_interval = save_weights_interval
        self.save_reconstruction_interval = save_reconstruction_interval
        self.verbose = verbose

        self.load_data_config = load_data_config
        self.vqvae_config = vqvae_config

        self.loader = LoadData(**self.load_data_config.__dict__)

        self.best_val_loss = 1e9


    def initialize_model(self, to_gpu=True):
        loading_bar('Inicializando modelo', 'red', '- initializing model -')
        model = VQVAE(**self.vqvae_config.__dict__)
        if to_gpu:
            model = model.to(self.device)

        return model

    def configure_tqdm(self, dataloader, epoch):
        t = tqdm(
            dataloader,
            desc=f"TRAINING [Epoch: {epoch+1}]",
            dynamic_ncols=True,
            colour="red",
        )
        return t

    def save_models(self, model, model_name, epoch, val_loss, path):
        if val_loss < self.best_val_loss:
            name = f"BestVal_{model_name}"
            save_path = os.path.join(path, name)
            torch.save(model.state_dict(), save_path)

        if epoch % self.save_weights_interval == 0:
            name = f"epoch_{epoch}_" + model_name
            save_path = os.path.join(path, name)
            torch.save(model.state_dict(), save_path)

    def run(self):
        print('-----------------------------')
        print('   STARTING VQVAE TRAINING   ')
        print('-----------------------------')

        # Loading data
        train_dataloader, val_dataloader = self.loader.get_train_dataloader()

        # Init model
        model = self.initialize_model()

        # Training configurations
        criterion = torch.nn.MSELoss()
        opt_vq = torch.optim.AdamW(model.parameters(), lr=3e-4)

        print('- training vqvae model -')

        all_train_loss = []
        all_val_loss = []

        for epoch in range(self.epochs):
            epoch_train_losses = []
            epoch_val_losses = []

            t = self.configure_tqdm(train_dataloader, epoch)
            model.train()
            for imgs, _ in t:
                imgs = imgs.to(device=self.device)
                decoded_images, min_indices, q_loss = model(imgs)

                rec_loss = criterion(imgs, decoded_images)
                vq_loss = rec_loss + q_loss

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)
                opt_vq.step()
                epoch_train_losses.append(vq_loss.cpu().detach().numpy())


            model.eval()
            with torch.no_grad():
                for val_imgs, _ in val_dataloader:
                    imgs = imgs.to(device=self.device)
                    decoded_images, min_indices, q_loss = model(imgs)

                    rec_loss = criterion(imgs, decoded_images)
                    vq_loss = rec_loss + q_loss
                    epoch_val_losses.append(vq_loss.cpu().detach().numpy())

            all_train_loss.append(np.mean(epoch_train_losses))
            all_val_loss.append(np.mean(epoch_val_losses))

            # Plotting the losses
            plot_losses(all_train_loss, all_val_loss, self.save_losses_path, "VQVAE")

            # Save the reconstruction images
            self.save_reconstruction(decoded_images, imgs, epoch)

            # Saving model checkpoints
            self.save_models(model,
                             'VQVAE',
                             epoch,
                             np.mean(epoch_val_losses),
                             self.save_model_path,
            )
            print("-")

            t.colour = 'green'
            t.colour = 'green'
            t.set_description("COMPLETO")
            t.set_postfix({"Status": "Completo"})
            t.refresh()
            t.close()
