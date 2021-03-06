
# Procgen Autoencoders

[![Python 3.6+](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![GitHub](https://img.shields.io/github/license/neuroevolution-ai/ProcgenAutoencoder)](https://github.com/neuroevolution-ai/ProcgenAutoencoder/blob/main/LICENSE)

![](./screenshots/render_example.png ) 



This repository was created in the internship Engineering Software Development WS2021 at the Karlsruhe Institute of Technology. 
The goal of the internship was to create a Gym environment for the Procgen game _**Heist**_, which outputs an encoding of an autoencoder as observation space. 

This repository contains the created gym environments in the `gym_autoencoder`folder  as well as scripts to train and create own gym environments for other procgen games.

## Installation
1) Clone the repository
2) Execute the following command in the source folder
```bash
pip install -e .
```

This repository is tested on this package versions:
-   python 3.8
-  opencv	4.5.0	
-  torchvision	0.8.1	
- torchinfo	0.0.5	
- procgen	0.10.4	
- pytorch	1.7.0	
- pyyaml	5.3.1
- matplotlib	3.3.2	

   
## Use Gym-Environments
After you installed your package , you can create an instance of the environment with:

```python
import gym
test_env = gym.make('gym_autoencoder:heist-auto-unpool-v0')
while(True):
    observation, rew, done, info = test_env.step(test_env.action_space.sample())
    test_env.render()
```
You should than be able to see the the heist representation from the Autoencoder. The Agent starts making random moves.

In this repository there are 6 evironments for the game heist, with 6 different Autoencoder architectures namely:

```
gym_autoencoder:heist-auto-unpool-v0
gym_autoencoder:heist-auto-maxpool-v0
gym_autoencoder:heist-auto-maxpool-big-v0
gym_autoencoder:heist-auto-no-bottleneck-v0
gym_autoencoder:heist-vae-alex-v0
gym_autoencoder:heist-vae-paper-v0
```
You can switch between CPU/GPU usage with the argument ```use_gpu```. Default value is True
 
 Example:
```python
test_env = gym.make('gym_autoencoder:heist-auto-unpool-v0',use_gpu=False)
```

The architectures used can be examined in the model_stubs folder. A short description can be found in the "Autoencoder Architectures" section.

**Important note:** There are 6 Gym environments available, but for further use only `auto-unpool-v0` and `vae-alex-v0` should be used, as the other autoencoders do not provide useful encodings.


## Train your own Autoencoder with own Dataset

This section describes how you can generate Data and Train your Autoencoder yourself. If you just want to use the predefined Gym-Environments you can skip reading this section.



### Generating Samples

For Data Generation use the files provided in the `data_generation` Folder. Running `sample_generator_heist_basic.py` will result in a _training.npy_ and _test.npy_ for the Procgen Game Heist.
For a balanced Dataset (Equal Number of Pictures with Keys/Doors and without Keys/Doors) use `sample_generator_heist.py` 

**Warning:**  `sample_generator_heist` needs ~20x time for Data Generation

### Training Autoencoders
After Generating samples run `autoencoder_training.py` Hyperparameters can be changed in the `training_hyperparameters.yaml` File. 

Training generates checkpoints if test loss gets better. Early Stopping is used (default=10)

**Notes:** Autoencoder will be saved in specified save directory. After Training Autoencoder will be evaluated on different Metrics.

### Evaluation Autoencoder

Speed of Autoencoder (GPU & CPU) will be tested with Batch Size 1 and 64. Model Architecture and Speed will 
be saved in a `summary.txt` File.
A Video will be created like featured below. Also Graph containing Train/Test Loss per Epoch will be saved.
If you want to make further evaluations feel free to use the generated `metrics.npy` and the model itself.

**Video Beispiel**

Left input Image, right generated image by the autoencoder. Prints also Loss-Function 
![Alt text](./screenshots/video_example.png?raw=true "Video Example") 
 
### Make Gym-Environment with trained Autoencoder

Create a new python file that inherits from `auto_basic_env.AutoencoderBasicEnv` You need the path to the trained model,
 the skeleton of the autoencoder and the dimension of the output vector. Orientate yourself on the existing environments. 
 Also add your class in the `/gym_autoencoder/__init__.py.`
 After that you should be able to run your autoencoder like descriped above.


## Autoencoder Architectures

Every Autoencoder inherit from `BaseAutoencoder.py` and should overwrite `encode`,`forward` and `loss_function` This work has implemented six different Autoenencoder for Usage. Feel free to extend this work :)

| Encoding Autoencoder | Autoencoder Name | Description | Encoding Time
| ------ | ---- | ---- | ---------------------|
 ![Alt text](./screenshots/Conv_Unpool_example.png)| `conv_unpool.py` | Autoencoder with Convolutional, Maxpooling and afterwards Unpooling Layers. Inspired by https://mi.eng.cam.ac.uk/projects/segnet/ (Little Smaller Version, than Segnet) | GPU: 1.130ms <br /> CPU: 2.673ms
| ![Alt text](./screenshots/VAE_Paper_example.png) _______________________| `VAE_Paper.py`  | Varitional-Autoencoder based on https://arxiv.org/pdf/1803.10122.pdf | GPU:0.797ms <br /> CPU:1.646ms
| ![Alt text](./screenshots/VAE_Alex_example.png) | `VAE_Alex.py` | Same Autoencoder as `VAE_Paper.py` , but Perceptional Loss from AlexNet is used | GPU: 0.792ms <br /> CPU:1.688ms
| ![Alt text](./screenshots/Conv_Maxpool_example.png) | `conv_maxpool_autoencoder.py`  | Autoencoder based on Convolutional Layers and Pooling Layers, followed by a Bottleneck Layer. Afterwards Transposed Convolutional Layers are used | GPU: 0.4374ms <br /> CPU: 1.211ms
| ![Alt text](./screenshots/Conv_MaxPool_big_example.png)  |  `conv_maxpool_big_autoencoder.py`| A deeper version of `conv_maxpool_autoencoder.py`  with more kernels used. | GPU 0.5217ms <br /> CPU: 1.0303ms
| ![Alt text](./screenshots/Conv_No_Bottleneck_example.png) |  `conv_no_bottleneck_autoencoder.py`| Autoencoder with Convolutional and Maxpooling Layer, but without any bottleneck layer | GPU 0.5253ms <br /> CPU: 1.8314ms
|


