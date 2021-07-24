# DualScriptASR
This work was benchmarked for the MUCS 2021 challenge (https://navana-tech.github.io/IS21SS-indicASRchallenge/challenge_details.html). This repository contains the codebase to repeat our results (https://arxiv.org/abs/2106.01400) on the blind test set of MUCS 2021

## Requirements 
- ESPNet 0.9.6
- Kaldi
- PyTorch 1.4 

## Pre-trained models
  * Subtask1 - https://drive.google.com/file/d/1LCnw0VnWgl5dFRwNhQ5cBYf8zzO1hXnY/view?usp=sharing
  * Subtask2 - https://drive.google.com/file/d/1bZdSAmR_trLQFAD1H_OJnfRhzFyHfVGk/view?usp=sharing

## Steps
Follow the below-given steps to repeat the results.
1. Make sure all the requirements are installed. Version mismatch may lead to an error while loading the pre-trained models.
2. Clone this repository
3. Relink espnet_root to the location where ESPnet is installed
4. Download the pre-trained models and place them in the root directory of this repository
5. Run 'bash run_blind_decoding.sh' 
6. 'run_blind_decoding.sh' will decode the blind data and create two submission files in the root directory. The submission files follow the format given in https://navana-tech.github.io/IS21SS-indicASRchallenge/btest.html 



