import importlib
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary as sf
from utils.NumpyDataLoader import NumpyDataLoader


class PerformanceEvaluation():
    def __init__(self, path, path_stub, path_test_data, model=None):
        self.path = path
        self.path_metrics = path + "metrics.npz"
        self.name = path.split("/")[-2]
        self.path_stub = path_stub
        if model is not None:
            self.model = model
        else:
            self.model = importlib.import_module(path_stub).Autoencoder()
            self.model.load_state_dict(torch.load(path + self.name + ".pt"))
        self.model.eval()
        self.test_data = NumpyDataLoader(path_test_data)
        self.metrics = np.load(self.path_metrics, allow_pickle=True)

    def toTensor(self, img):
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img / 255
        return img

    def test_speed_batch(self, batch_size=64):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, pin_memory=True, shuffle=True)
        total_gpu_time = 0
        total_cpu_time = 0
        with torch.no_grad():
            self.model.cuda()
            for batch in test_loader:
                batch = batch.cuda() / 255
                start = time.time()
                self.model(batch)
                end = time.time()
                total_gpu_time += (end - start)
            self.model.cpu()
            for batch in test_loader:
                batch = batch / 255
                start = time.time()
                self.model(batch)
                end = time.time()
                total_cpu_time += (end - start)
        total_gpu_time = total_gpu_time / len(self.test_data) * 1000
        total_cpu_time = total_cpu_time / len(self.test_data) * 1000
        output_str = self.format_performance(batch_size, total_cpu_time, total_gpu_time)
        print(output_str)
        return output_str, (total_cpu_time, total_gpu_time)

    def test_speed_batch_encoder(self, batch_size=64):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, pin_memory=True, shuffle=True)
        total_gpu_time = 0
        total_cpu_time = 0
        with torch.no_grad():
            self.model.cuda()
            for batch in test_loader:
                batch = batch.cuda() / 255
                start = time.time()
                self.model.encode(batch)
                end = time.time()
                total_gpu_time += (end - start)
            self.model.cpu()
            for batch in test_loader:
                batch = batch / 255
                start = time.time()
                self.model.encode(batch)
                end = time.time()
                total_cpu_time += (end - start)
        total_gpu_time = total_gpu_time / len(self.test_data) * 1000
        total_cpu_time = total_cpu_time / len(self.test_data) * 1000
        output_str = self.format_performance(batch_size, total_cpu_time, total_gpu_time)
        print(output_str)
        return output_str, (total_cpu_time, total_gpu_time)

    def format_performance(self, batch_size, total_cpu_time, total_gpu_time):
        output_str = (f'With Batch Size ={batch_size}\n')
        output_str += (f'The average GPU Time per sample is {total_gpu_time} milliseconds.') + " \n"
        output_str += (f'The average CPU Time per sample is {total_cpu_time} milliseconds.') + "\n"
        return output_str

    def write_summary(self):
        # summary(self.model,(3,64,64))
        f = open(self.path + "summary.txt", "w")
        f.write('Summary Model\n')
        self.model.cuda()
        f.write(str(sf(self.model,input_size=(1,3,64,64))))
        f.write(" \n")
        mini_batch, batch = 1, 64
        output_str, _ = self.test_speed_batch(mini_batch)
        f.write('The whole model needs \n')
        f.write(output_str)
        output_str, _ = self.test_speed_batch(batch)
        f.write(output_str)
        f.write("----------------------------------------------------------------\n")
        f.write("The encoder part needs \n")
        output_str, _ = self.test_speed_batch_encoder(mini_batch)
        f.write(output_str)
        output_str, _ = self.test_speed_batch_encoder(batch)
        f.write(output_str)
        f.close()

    def tranpose(self, img):
        return np.transpose(img, (2, 1, 0))

    def make_video(self):
        self.model.cuda()
        img_array = []
        for img in self.test_data:
            img_tensor = self.toTensor(img).cuda()
            result = self.model(img_tensor)
            loss = self.model.loss_function(result, input=img_tensor).item()
            loss = "{:.7f}".format(round(loss, 8))
            result = result[0].cpu().detach().numpy() * 255
            result = result[0].astype(np.uint8)
            concat = np.concatenate((self.tranpose(img), self.tranpose(result)), axis=1)
            bigger = cv2.resize(concat, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bigger, loss, (0, 25), font, 1, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.putText(bigger, loss, (0, 25), font, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            bigger = cv2.cvtColor(bigger, cv2.COLOR_BGR2RGB)
            cv2.imshow("Result", bigger)
            cv2.waitKey(1)
            img_array.append(bigger)

        width, height, _ = img_array[0].shape
        out = cv2.VideoWriter(self.path + 'video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 4, (height, width))
        for img in img_array:
            out.write(img)
        cv2.destroyAllWindows()
        out.release()

    def printMetrics(self):
        plt.figure(12)
        plt.plot(self.metrics['epoch_train_loss'], "-o", markersize=3, color="blue", label="Train Loss")
        plt.plot(self.metrics['epoch_test_loss'], "-o", markersize=3, color="red", label="Test Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochen")
        plt.ylabel("Fehler")
        plt.grid()
        plt.savefig(self.path + "metrics.pdf")

    def printNTimes(self, n):

        def eval_and_print(img, ax, index):
            self.model.cuda()
            result = self.model(self.toTensor(img).cuda())
            result = result[0].cpu().detach().numpy()
            result = self.tranpose(result[0])
            ax[0, index].axis("off")
            ax[1, index].axis("off")
            ax[0, index].imshow(result)
            ax[1, index].imshow(self.tranpose(img))

        def printImages(k):
            fig, axs = plt.subplots(2, 5, figsize=(18, 8))
            plt.subplots_adjust(top=0.981,
                                bottom=0.019,
                                left=0.008,
                                right=0.992,
                                hspace=0.0,
                                wspace=0.044)

            for i in range(5):
                next_i = random.randrange(0, len(self.test_data))
                eval_and_print(self.test_data[next_i], axs, i)
            plt.savefig(self.path + 'images' + str(k) + '.jpg')

        for i in range(n):
            plt.figure(i)
            printImages(i)

    def evaluation(self):
        self.make_video()
        self.write_summary()
        self.printMetrics()
        self.printNTimes(1)


if __name__ == '__main__':
    path = "./trained_models/Testing/"
    path_model_stub = "model_stubs.conv_unpool"
    test_samples_filepath = "data/heist/test_samples_memory_noBack_balanced2.npy"

    perforcman = PerformanceEvaluation(path, path_model_stub, test_samples_filepath)
    perforcman.evaluation()
