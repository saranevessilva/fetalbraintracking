# SVRTK : SVR reconstruction based on MIRTK and CNN-based processing for fetal MRI
#
# Copyright 2018-2020 King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# see the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md

# ==================================================================================================================== #
#
#  TRAIN Localisation Network with 3D images
#
# ==================================================================================================================== #
N_epochs = 100
I_size = 128
N_classes = 2

# # # Prepare arguments

args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                      batch_size=2,
                                      lr=0.002,
                                      crop_height=I_size,
                                      crop_width=I_size,
                                      crop_depth=I_size,
                                      validation_steps=8,
                                      lamda=10,
                                      training=True,
                                      testing=False,
<<<<<<< HEAD
                                      running=True,
                                      root_dir='/home/sn21/fetal-brain-tracking/Gadgetron/',
                                      csv_dir='/home/sn21/fetal-brain-tracking/Gadgetron/files/',
=======
                                      running=False,
                                      root_dir='/home/sn21/localiser',
                                      csv_dir='/home/sn21/localiser/files/',
>>>>>>> 007a79857133678e77b3da8b039d089e3dbc0a3f
                                      train_csv='data_localisation_1-label-brain_uterus_train-2022-10-25.csv',
                                      valid_csv='data_localisation_1-label-brain_uterus_valid-2022-10-25.csv',
                                      test_csv='run-21-10-2022.csv',
                                      run_csv='run-21-10-2022.csv',
                                      results_dir='/home/sn21/localiser/results/2022-10-25',
                                      checkpoint_dir='/home/sn21/localiser/current-checkpoints/2022-10-25/',
                                      exp_name='Loc_3D',
                                      task_net='unet_3D',
                                      n_classes=N_classes)


args.gpu_ids = [0]

# RUN training

if args.training:
    print("Training")
    model = md.LocalisationNetwork3DMultipleLabels(args)

    # Run train
    ####################
    losses_train = model.train(args,0)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')

# TEST to compare with the ground truth results

if args.testing:
    print("Testing")
    model = md.LocalisationNetwork3DMultipleLabels(args)

    # Run inference
    ####################
    model.test(args,1)

# RUN with empty masks - to generate new ones (practical application)

if args.running:
    print("Running")
    model = md.LocalisationNetwork3DMultipleLabels(args)

    # Run inference
    ####################
    model.run(args, 1)

    print("Localisation completed.")

