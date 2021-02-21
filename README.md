
# Procgen Autoencoders

`print(Hello)`


## Use Autoencoders

1 ) Install Autoencoder using
```bash
cd gym_autoencoder
pip install -e .
```
2 ) After you installed your package , you can create an instance of the environment with:

```python
import gym_autoencoder
env = gym.make('auto-unpool-v0')

```

## Train your own Autoencoder with own Dataset

This section describes how you can generate Data and Train your Autoencoder yourself. If you just want to use the predefined Gym-Environments skip to



### Generating Samples

For Data Generation use the files provided in the data_generation Folder. Running `sample_generator_heist_basic.py` will result in a _training.npy_ and _test.npy_ for the Procgen Game Heist.
For a balanced Dataset (Equal Number of Pictures with Keys/Doors and without Keys/Doors) use `sample_generator_heist.py` 

**Warning:**  `sample_generator_heist` needs ~20x time for Data Generation

### Training Autoencoders
After Generating samples run `autoencoder_training.py` Hyperparameters can be changed in the `training_hyperparameters.yaml` File. 

Training generates Checkpoints if test loss gets better. Early Stopping is used (default=10)

**Notes:** Autoencoder will be saved in specified save directory. After Training Autoencoder will be evaluated on different Metrics.

### Evaluation Autoencoder

Speed of Autoencoder (GPU & CPU) will be tested with Batch Size 1 and 64. odel Architecture and Speed will 
be saved in a `summary.txt` File.
A Video will be created like featured below. Also Graph containing Train/Test Loss per Epoch will be saved.
I you want to make further evaluations feel free to use the generated `metrics.npy` and the model itself.

**Video Beispiel**

Left input Image, right generated image by the autoencoder. Prints also Loss-Function 
![Alt text](autoencoder-procgen/screenshots/video_example.png?raw=true "Video Example") 
 
 


## Autoencoder Architectures

Every Autoencoder inherit from `BaseAutoencoder.py` and should overwrite `encode`,`forward` and `loss_function` This work has implemented six different Autoenencoder for Usage. Feel free to extend this work :)

| First Header  | Autoencoder Name | Description
| ------------- | ------------- | --------- |
| Content Cell  | `conv_maxpool_autoencoder.py`  | Autoencoder based on Convolutional Layers and Pooling Layers, afterwords an Bottleneck Layer. Afterwards Transposed Convolutional Layers are used
| Content Cell  |  `conv_maxpool_big_autoencoder.py`| A deeper version of `conv_maxpool_autoencoder.py`  with more kernels used.
| Content Cell  |  `conv_no_bottleneck_autoencoder.py`| Autoencoder with Convolutional and Maxpooling Layer, but without any bottleneck layer
| Content Cell  | `conv_unpool.py` | Autoencoder with Convolutional, Maxpooling and afterwards Unpooling Layers. Inspired by https://mi.eng.cam.ac.uk/projects/segnet/ (Little Smaller Version, than Segnet)
| Content Cell | `VAE_Paper.py` | Varitional-Autoencoder based on https://arxiv.org/pdf/1803.10122.pdf
| Concnten Cell | `VAE_Alex.py` | Same Autoencoder as `VAE_Paper.py` , but Perceptional Loss from AlexNet is used



## Training Autoencoders
