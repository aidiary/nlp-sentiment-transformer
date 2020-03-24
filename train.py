from comet_ml import Experiment

import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import create_dataloader
from model import TransformerClassification

experiment = Experiment(project_name="nlp_sentiment_transformer", auto_metric_logging=False)


def main():
    os.makedirs('checkpoints', exist_ok=True)

    # load data
    train_dl, val_dl, test_dl, TEXT = create_dataloader()
    dataloaders_dict = {'train': train_dl, 'val': val_dl}

    # load model
    net = TransformerClassification(TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)
    net.train()
    net.net3_1.apply(weights_init)
    net.net3_2.apply(weights_init)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    learning_rate = 2e-5
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # training
    num_epochs = 30
    net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

    # save trained model
    torch.save(net_trained.state_dict(), 'checkpoints/model.pt')


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('***', device)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0.0

            for batch in dataloaders_dict[phase]:
                inputs = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                optimizer.zero_grad()

                # trainの場合のみ逆伝搬のgradを保存
                with torch.set_grad_enabled(phase == 'train'):
                    # <pad>の単語IDは1
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            # logging by comet.ml
            if phase == 'train':
                experiment.log_metric('train_loss', epoch_loss, step=epoch)
                experiment.log_metric('train_acc', epoch_acc.item(), step=epoch)
            else:
                experiment.log_metric('valid_loss', epoch_loss, step=epoch)
                experiment.log_metric('val_acc', epoch_acc.item(), step=epoch)

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

    return net


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    main()
