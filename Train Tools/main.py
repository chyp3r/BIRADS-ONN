import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from birads_onn import *
from data_loader import *

def main():
    # Hyperparameters
    IN_CHANNELS = 3
    BIRADS_NUM_CLASSES = 3
    COMPOSITION_NUM_CLASSES = 4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 48
    NUM_EPOCHS = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    SAVE_PATH = 'checkpoint.pth.tar'

    train_loader =train_loader_f 
    validation_loader =test_loader_f

    # Model
    model = VGG_ONN(VGG16_N, in_channels=IN_CHANNELS, num_classes=3)    
    model = model.to(device=device)
     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer = CustomTrainer(model, optimizer, criterion, device)
    early_stopper = EarlyStopper(patience=NUM_EPOCHS//6, min_delta=0)
    checkpoint = ValidationLossCheckpoint(path=SAVE_PATH)

    trainer.train(NUM_EPOCHS, train_loader=train_loader, validation_loader=validation_loader, 
                  early_stopper=early_stopper, checkpoint=checkpoint)

if __name__ == "__main__":    
    main()