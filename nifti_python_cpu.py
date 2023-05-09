import numpy as np
import gadgetron
import ismrmrd
import logging
import time
import io
import os

import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ismrmrd_to_nifti
import sys

# from python_version import extract_ismrmrd_parameters_from_headers as param, flip_image as fi, set_nii_hdr as tools
import nibabel as nib

from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md


def IsmrmrdToNiftiGadget(connection):
    logging.info("IsmrmrdToNifti, process ... ")
    start = time.time()

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    ncoils = 1

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z

    # Initialise a storage array
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

    ninstances = nslices * nreps
    print("Number of instances ", ninstances)

    im = np.zeros((eNx, eNy, nslices), dtype=np.complex64)

    for acquisition in connection:
        image = np.abs(acquisition.data.transpose(3, 2, 1, 0))

        ndim = image.shape
        # print("IsmrmrdToNifti, receiving image array of shape ", ndim)
        # print("IsmrmrdToNifti, receiving image head :", acquisition)

        # # Create parameters for set_nii_hdr et xform_mat
        # h = param.extract_ismrmrd_parameters_from_headers(acquisition, hdr)
        # print("IsmrmrdToNifti, computed Nifti parameters : ", h)

        # Get crop image, flip and rotate to match with true Nifti image
        img = image[:, :, :, 0].transpose(1, 0, 2)

        # Stuff into the buffer
        slice = acquisition.slice
        im[:, :, slice] = np.squeeze(img[:, :, 0])
        # Still need to do some work here to separate the repetitions

    # print(im)
    # print(im.shape)

    # Create nii struct based on img
    nii = nib.Nifti1Image(np.abs(im), np.eye(4))

    # Save image in nifti format
    output_path = os.path.join('/tmp/gadgetron/nifti_manon_gadgetron.nii.gz')
    # output_path = os.path.join('/tmp/gadgetron/nifti_manon_gadgetron'+str(start)+'.nii.gz')
    # output_path = os.path.join('/home/sns/fetal-brain-track/run/nifti_manon_gadgetron.nii.gz')
    nib.save(nii, output_path)
    print(output_path)

    print("Nifti well saved")

    # print data infos
    print("-------------------------------------------------------------------------")

    logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

    # for s in range(nslices):
    #     plt.imshow(np.squeeze(np.abs(im[:, :, s, 0])), cmap="gray")
    #     plt.show()
    #
    # for count, value in enumerate(im.shape):
    #     if count >= 3:
    #         for i in range(0, im.shape[3]):
    #             new_3D = np.squeeze(im[:, :, :, i])
    #             print("New shape:", new_3D.shape)

    # ==================================================================================================== #
    #
    #  TRAIN Localisation Network with 3D images
    #
    # ==================================================================================================== #

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
                                          training=False,
                                          testing=False,
                                          running=True,
                                          root_dir='/tmp/gadgetron',
                                          csv_dir='/tmp/gadgetron/files/',
                                          train_csv=
                                          'data_localisation_1-label-brain_uterus_train.csv',
                                          valid_csv=
                                          'data_localisation_1-label-brain_uterus_valid.csv',
                                          test_csv=
                                          'data_localisation_1-label-brain_uterus_run.csv',
                                          run_csv=
                                          'data_localisation_1-label-brain_uterus_run.csv',
                                          results_dir='/tmp/gadgetron/results/',
                                          checkpoint_dir='/tmp/gadgetron/checkpoints/',
                                          exp_name='Loc_3D',
                                          task_net='unet_3D',
                                          n_classes=N_classes)

    # for acquisition in connection:

    args.gpu_ids = [0]

    # RUN with empty masks - to generate new ones (practical application)

    if args.running:
        print("Running")
        model = md.LocalisationNetwork3DMultipleLabels(args)

        # Run inference
        ####################
        model.run(args, 1)

        print("Localisation completed.")

