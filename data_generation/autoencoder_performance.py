import importlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class PerformanceEvaluation():
    def __init__(self, path, path_model, path_stub, path_metrics, path_test_data, model=None):
        self.path = path
        self.path_metrics = path_metrics
        self.path_stub = path_stub
        if model is not None:
            self.model = model
        else:
            self.model = importlib.import_module(path_stub).Autoencoder()
            self.model.load_state_dict(torch.load(path_model))
        self.model.eval()
        self.test_data = NumpyDataLoader(path_test_data)
        self.metrics = np.load(path_metrics, allow_pickle=True)

    def tranpose(self, img):
        img = img  # unnormalize
        return np.transpose(img, (2, 1, 0))

    def toTensor(self, img):
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img / 255
        img = img.permute(0, 3, 2, 1)
        return img

    def eval_and_print(self, img, ax, index):
        result = self.model(self.toTensor(img))
        result = result[0].detach().numpy()
        result = self.tranpose(result[0])
        ax[0, index].axis("off")
        ax[1, index].axis("off")
        ax[0, index].imshow(result)
        ax[1, index].imshow(img)

    def printImages(self, k):
        fig, axs = plt.subplots(2, 5, figsize=(18, 8))
        plt.subplots_adjust(top=0.981,
                            bottom=0.019,
                            left=0.008,
                            right=0.992,
                            hspace=0.0,
                            wspace=0.044)

        for i in range(5):
            next_i = random.randrange(0, len(self.test_data))
            self.eval_and_print(self.test_data[next_i], axs, i)
        plt.savefig(self.path + 'images' + str(k) + '.jpg')

    def printMetrics(self):
        plt.figure(12)
        plt.plot(self.metrics['epoch_train_loss'], "b")
        plt.plot(self.metrics['epoch_test_loss'], "r")
        plt.xlabel("Epochen")
        plt.ylabel("Fehler")
        plt.savefig(self.path + "metrics.pdf")

    def printNTimes(self, n):
        for i in range(n):
            plt.figure(i)
            self.printImages(i)


class NumpyDataLoader(Dataset):
    def __init__(self, path):
        self.datas = np.load(path)
        self.length = self.datas.shape[0]
        print(self.datas.shape)
        print(self.datas.dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.datas[idx]


if __name__ == '__main__':
    path = "./trained_models/BasicC/"
    path_metrics = "./trained_models/BasicC/metrics.npz"
    path_model = 'trained_models/BasicC/BasicC2.pt'
    path_model_stub = "model_stubs.thesis_autoencoder"
    test_samples_filepath = "./test_samples_memory.npy"

    perforcman = PerformanceEvaluation(path, path_model, path_model_stub, path_metrics, test_samples_filepath)
    perforcman.printNTimes(10)
    perforcman.printMetrics()
    plt.plot()
