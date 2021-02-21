import argparse
import importlib
import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from utils.torch_summary import summary
from torchvision import transforms

import performance_eval as perf
from utils.NumpyDataLoader import NumpyDataLoader

with open("training_hyperparameter.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser(description='Parameter for Model and Model-Name')
parser.add_argument('--name', type=str, default="ModelNew",
                    help='Name Model')
parser.add_argument('--model_path', type=str, default="model_stubs.experimental.autoencoder_unpool",
                    help='Modul Stub')
parser.add_argument('--num', type=int, default="32",
                    help='latent_dim')

args = parser.parse_args()
config["logging_params"]['name'] = args.name
config['data_params']['model_path'] = args.model_path


def create_folder_structure(directory, name_model):
    '''
    Creates a Folder Structure at the directory path given with the name_model
    For example giving the arguments  path/to/x/ , TestAutoencoder will result in the creation of
    the folder
    path/to/x/TestAutoencoder
    path/to/x/TestAutoencoder/checkpoints

    The function will return above Strings

    :param directory: String of directory path
    :param name_model: Name of the Folder that will be created
    :return: Two Strings, first relative path where the model is
                          second relative checkpoint path
    '''
    model_path = directory + name_model + "/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_path = directory + name_model + "/checkpoints"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    return model_path, checkpoint_path


def load_data():
    train_dir = config['data_params']['training_filepath']
    test_dir = config['data_params']['test_filepath']
    training_data = NumpyDataLoader(train_dir)
    test_data = NumpyDataLoader(test_dir)
    config["num_train"] = len(training_data)
    config["num_test"] = len(test_data)
    batch_size = config['training_params']['batch_size']
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, pin_memory=True, shuffle=True)
    return train_loader, test_loader


def load_model():
    model_path = config['data_params']['model_path']
    autoencoder = importlib.import_module(model_path)
    model = autoencoder.Autoencoder()
    model.cuda()
    return model


def setup_optimizer(model):
    lr, weight_decay = config['training_params']['LR'], config['training_params']['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    return optimizer


def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    batch_size = config['training_params']['batch_size']
    number_samples = config["num_train"]
    log_interval = len(train_loader) // log_interval
    if log_interval == 0:
        log_interval = 1

    for batch_index, data in enumerate(train_loader):
        img = data.float().cuda()  / 255
        output = model(img)
        output.append(batch_size/number_samples)
        loss = model.loss_function(output, input=img)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

        if batch_index % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train Epoch:{}, Loss:{:.4f}'.format(epoch, float(train_loss)))
    torch.cuda.empty_cache()
    return train_loss


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    batch_size = config['training_params']['batch_size']
    number_samples = config["num_test"]
    with torch.no_grad():
        for batch in test_loader:
            img = batch.float().cuda() / 255
            output = model(img)
            output.append(batch_size/number_samples)
            test_loss += model.loss_function(output, input=img)
    test_loss /= len(test_loader)

    print('Test Epoch:{}, Loss:{:.4f}'.format(epoch, float(test_loss)))
    return test_loss


def epoch_steps(model, train_loader, test_loader, optimizer, directory_checkpoint):
    epochs, log_interval = config['training_params']['epochs'], \
                           config['training_params']['log_interval']
    early_stopping_num = config['training_params']['early_stopping']

    epoch_train_loss = []
    epoch_test_loss = []
    current_best = float("inf")
    epochs_no_improve = 0
    current_best_model = None
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, log_interval)
        test_loss = test(model, test_loader, epoch)
        epoch_train_loss.append(train_loss)
        epoch_test_loss.append(test_loss)
        if test_loss < current_best:
            epochs_no_improve = 0
            formattet_loss = "{:.4f}".format(float(test_loss))
            current_best_model = model
            torch.save(model.state_dict(), directory_checkpoint + "/E" + str(epoch) + "L" + formattet_loss + ".pt")
            current_best = test_loss
        else:
            epochs_no_improve += 1
        if epochs_no_improve > early_stopping_num:
            print("Early Stopping")
            break

    return epoch_train_loss, epoch_test_loss, current_best_model


def save_model(model, directory_model, name, epoch_train_loss, epoch_test_loss):
    np.savez(directory_model + "/metrics", epoch_train_loss=epoch_train_loss, epoch_test_loss=epoch_test_loss)

    torch.save(model.state_dict(), directory_model + "/" + name + ".pt")

    path_model = directory_model + "/" + name + ".pt"
    path_model = path_model[2:]
    return path_model


def main():
    save_dir, name = config['logging_params']['save_dir'], config['logging_params']['name']
    directory_model, directory_checkpoint = create_folder_structure(save_dir, name)
    train_loader, test_loader = load_data()
    model = load_model()
    optimizer = setup_optimizer(model)
#    print(summary(model, (3, 64, 64)))
    epoch_train_loss, epoch_test_loss, current_best_model = epoch_steps(model, train_loader, test_loader, optimizer,
                                                                        directory_checkpoint)
    path_model = save_model(current_best_model, directory_model, name, epoch_train_loss, epoch_test_loss)

    perfor = perf.PerformanceEvaluation(directory_model, config['data_params']['model_path'],
                                        config['data_params']['test_filepath'], current_best_model)
    perfor.evaluation()


if __name__ == '__main__':
    main()
