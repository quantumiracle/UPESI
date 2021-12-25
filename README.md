# UPESI

Code for paper [Not Only Domain Randomization: Universal Policy with Embedding System Identification](https://arxiv.org/abs/2109.13438).

 ## Installation
 This repo use the same environment as [another repo](https://github.com/quantumiracle/Robotic_Door_Opening_with_Tactile_Simulation). Please follow the steps there to install our modified version of robosuite.
 
 ## Training Procedure
 For the universal policy (UP) with embedding system identification (ESI), we use the following commands.

 First go to the directory:
 ```bash
  cd dynamics_predict
 ```
1. Collect training and testing dataset
  ```bash
  python train_dynamics.py --collect_train_data --env Env_NAME
  python train_dynamics.py --collect_test_data --env Env_NAME
  ```
2. Normailize data
 Run
 ```bash
  cd ../data/dynamics_data
  jupyter notebook
 ```
  and open ```data_process_*ENV_NAME*.ipynb``` and go through each cell.

3. Train dynamics embedding (encoder, decoder and dynamics prediction model)

 Back to the terminal in ```dynamics_predict/```.

 Run the following to lauch training,
  ```bash
  python train_dynamics.py --train_embedding --env Env_NAME
  ```
  and use launch ```tensorboard --logdir runs``` to monitor the training process. 

4. Test learned encoder and dynamics predictor
 Test the preformance of learned encoder and dynamics predictor by applying them in ESI on collected test data:
  ```bash
  jupyter notebook
  ```
  and open ```test_dynamics_*ENV_NAME*.ipynb``` and go through each cell, including a Bayesian optimization (BO) process.

5. Train UP
  ```bash
  cd ..
  python train.py --train --env *ENV_NAME*dynamics --process NUM 
  ```
  Select the encoder-decoder type in `./environment/*ENV_NAME*dynamics.py` to match with the one used in `./dynamics_predict/train_dynamics.py`.

6. Test ESI with UP against other methods
  ```bash
  cd dynamics_predict
  python compare_methods_*ENV_NAME*.py
  ```
