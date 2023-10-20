import torch
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm
import time

def loading_bar(mensagem='Carregando', colour='red', print_text='- loading pretrained model -'):
    print('\n')
    print(print_text)
    # Número total de itens a serem carregados
    total_itens = 100

    # Inicialize a barra de progresso com 0% (azul)
    barra_progresso = tqdm(total=total_itens, desc=mensagem, bar_format="{desc}: {percentage:3.0f}% {bar}", colour=colour)

    # Simule o carregamento de dados
    for i in range(total_itens):
        # Simule o carregamento de um item
        time.sleep(0.018)  # Simule uma carga de dados
        barra_progresso.update(1)  # Atualize a barra de progresso em 1 unidade

    # Carregamento completo, altere a barra para verde e exiba a mensagem
    barra_progresso.colour = 'green'
    barra_progresso.set_description("COMPLETO")
    barra_progresso.set_postfix({"Status": "Completo"})
    barra_progresso.refresh()
    barra_progresso.close()

    print("\033[32mCarregamento completo!\033[0m")
    print('\n')

def plot_losses(train_loss, val_loss, path, model_name):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train loss", color="tab:blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation loss", color="tab:orange", linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(
        f"{model_name}: training and validation loss", fontsize=16, fontweight="bold"
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    # Grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Spines (border) color
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    # Tick width
    ax.tick_params(width=0.5)

    # Background color
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()

    save_path = os.path.join(path, model_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
