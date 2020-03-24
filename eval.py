from comet_ml import ExistingExperiment
import matplotlib.pyplot as plt
import torch
from data import create_dataloader
from model import TransformerClassification

experiment = ExistingExperiment(previous_experiment='b8d5b06e99484f8a93dd0d84f8a36f3e')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    _, _, test_dl, TEXT = create_dataloader()

    # load model
    net = TransformerClassification(TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)
    net.load_state_dict(torch.load('checkpoints/model.pt'))
    net.to(device)

    epoch_corrects = 0
    for batch in test_dl:
        inputs = batch.Text[0].to(device)
        labels = batch.Label.to(device)

        with torch.set_grad_enabled(False):
            input_pad = 1
            input_mask = (inputs != input_pad)

            outputs, _, _ = net(inputs, input_mask)
            _, preds = torch.max(outputs, 1)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
    print('***', epoch_acc.item())


if __name__ == "__main__":
    main()
