import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src import model
from src import dataloader


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super().__init__()
        self.m = m

    def forward(self, y1, y2, d):
        loss = 0
        for i in range(y1.shape[0]):
            euc_dist = torch.nn.functional.pairwise_distance(y1[i], y2[i])

            if d[i] == 0:
                loss += torch.mean(torch.pow(euc_dist, 2))
            else:
                delta = self.m - euc_dist
                delta = torch.clamp(delta, min=0.0, max=None)
                loss += torch.mean(torch.pow(delta, 2)) 

        return loss / y1.shape[0]


def my_collate(batch):
    return torch.utils.data.dataloader.default_collate(list(filter(lambda x:x is not None, batch)))


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.AudioCNN([65, 69, 2]).to(self.device)
        self.train_loader = torch.utils.data.DataLoader(
            dataloader.SSLAudioDataset('../sound-spaces/data', 'replica', 44100, 'train'),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=my_collate
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataloader.SSLAudioDataset('../sound-spaces/data', 'replica', 44100, 'val'),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=my_collate
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def loss_func(self, outputs, labels):
        contrastive_loss =ContrastiveLoss()(outputs['contrastive_embedding'][0], outputs['contrastive_embedding'][1], labels['contrastive_label'])
        angle_loss = torch.nn.CrossEntropyLoss()(outputs['angle_label'], labels['angle_label'].to(self.device))
        eu_dist_loss = torch.nn.BCEWithLogitsLoss()(outputs['eu_dist_label'][:, 0], labels['eu_dist_label'].to(self.device))
        return contrastive_loss + angle_loss + eu_dist_loss

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0
        for i, data in enumerate(self.train_loader):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            #Make predictions for this pass
            outputs = self.model.forward(data['spec_1'].to(self.device), data['spec_2'].to(self.device))

            # Compute the loss and its gradients
            loss = self.loss_func(outputs, data)
            loss.backward()

            self.optimizer.step()

             # Gather data and report
            running_loss += loss.item()
        return running_loss / (i + 1)

    def train_epochs(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 1000
        #Performing validation by checking the relative loss
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
            #Make sure gradient tracking is on, and do a pass over the data
            self.model.train()
            avg_loss = self.train_one_epoch(epoch_number, writer)
            #No gradients while reporting
            self.model.eval()
            #Calculating validation loss
            running_vloss = 0.0
            for i, vdata in enumerate(self.train_loader):
                #Make predictions for this pass
                voutputs = self.model.forward(vdata['spec_1'].to(self.device), vdata['spec_2'].to(self.device))
                # Compute the loss and its gradients
                vloss = self.loss_func(voutputs, vdata)

                # Gather data and report
                running_vloss += vloss.item()
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            
            # Log the running loss averaged per batch for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
            { 'Training' : avg_loss, 'Validation' : avg_vloss },
            epoch_number + 1)
            writer.flush()
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}'.format(best_vloss)
                self.model.save_backbone(model_path)

            epoch_number += 1


trainer = Trainer()
trainer.train_epochs()