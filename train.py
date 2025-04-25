import os
import argparse
import numpy as np
import pandas as pd
import net
import torch
from tqdm import tqdm
from process_data import TitanicDataset, collate_fn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import *


def train(params, model, train_features, val_features):
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.zero_grad()

    train_dataloader = DataLoader(train_features, batch_size=params.train_batch_size,
                            shuffle=True, drop_last=True, )


    for epoch in range(params.num_epochs):
        model.zero_grad()
        for _, features, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{params.num_epochs}"):
            model.train()   # Switch to the train mode

            # Move inputs to the same device of our model.
            features = features.clone().detach().requires_grad_(True).to(params.device)
            labels = labels.clone().detach().requires_grad_(True).to(params.device)

            outputs = model(features)
            loss = net.loss_fn(outputs, labels)

            # Update the parameters of the model
            loss.backward()
            optimizer.step()

            # The gradient of the model should be zero before next batch
            model.zero_grad()
        
        evaluate(params, model, val_features)
        #logging.info('Epoch: {:3} | Loss: {:8.4f} | ac: {:5.2f}'.format(
        #        epoch + 1, metrics_mean['loss'], metrics_mean['accuracy']))
        os.makedirs(os.path.dirname(os.path.join(params.model_dir, 'net.pt')), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(params.model_dir, 'net.pt')) # save the model




def evaluate(params, model, val_features):
    # set model to evaluation mode
    model.eval()

    metrics = net.metrics
    test_dataloader = DataLoader(val_features, batch_size=params.test_batch_size)

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _, features, labels in tqdm(test_dataloader, desc=f"evaluating"):
        # compute model output
        outputs = model(features)
        loss = net.loss_fn(outputs, labels)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        # outputs = outputs.data.cpu().numpy()
        # labels = labels.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](outputs, labels)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)


def test(params, model, test_features):
    # set model to evaluation mode
    model.eval()

    test_dataloader = DataLoader(test_features, batch_size=params.test_batch_size)
    PassengerId = []
    Survived = []

    # compute metrics over the dataset
    for ids, features, _ in test_dataloader:
        # compute model output
        outputs = model(features)
        outputs = (outputs > 0.5).int()
        PassengerId.append(ids.view(-1))
        Survived.append(outputs.view(-1))
    
    PassengerId = torch.concat(PassengerId, dim=0).detach().numpy()
    Survived = torch.concat(Survived, dim=0).detach().numpy()

    pred = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Survived})
    pred.to_csv('./data/processed/pred.csv', index=False)
    logging.info('test finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', default='model/params.json', help="params json path")
    parser.add_argument('--test', help='Test on the testset.', action='store_true')
    args = parser.parse_args()

    params = Params(args.params_path)

    # Set the logger
    set_logger(os.path.join(params.model_dir, 'train.log'))

    # Specify the device. If you has a GPU, the training process will be accelerated.
    device = torch.device('cuda:{}'.format(params.gpu) if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Load the datasets
    train_features = TitanicDataset(params, './data/processed/train.csv')
    val_features = TitanicDataset(params, './data/processed/test.csv')
    test_features = TitanicDataset(params, './data/origin/test.csv')

    model = net.Net(params)
    model = model.to(params.device)   # move the model into GPU if GPU is available.

    if not args.test:
        train(params, model, train_features, val_features)
    else:
        model.load_state_dict(torch.load(os.path.join(params.model_dir, 'net.pt')))
        test(params, model, test_features)


if __name__ == '__main__':
    main()