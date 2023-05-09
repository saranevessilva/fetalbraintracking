import numpy as np
import gadgetron
import ismrmrd
import logging
import time
import io
import os
from datetime import datetime

import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ismrmrd_to_nifti
import sys

# from python_version import extract_ismrmrd_parameters_from_headers as param, flip_image as fi, set_nii_hdr as tools
import nibabel as nib

import src.utils as utils
from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def send_reconstructed_images_wcm(connection, data_array, rotx, roty, rotz, cmx, cmy, cmz, acq_header):
    # this function sends the reconstructed images with centre-of-mass stored in the image header
    # the fonction creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

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

    dims = data_array.shape

    # print(acq_header)

    # base_header = acq_header

    base_header = ismrmrd.ImageHeader()
    base_header.version = 2
    # ndims_image = (dims[0], dims[1], dims[2], dims[3])
    ndims_image = (dims[0], dims[1], dims[2])
    base_header.channels = ncoils  # The coils have already been combined
    base_header.matrix_size = (data_array.shape[0], data_array.shape[1], data_array.shape[2])
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.image_index = 0
    base_header.image_series_index = 0
    base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    base_header.repetition = acq_header.repetition

    base_header.user_float[0] = rotx
    base_header.user_float[1] = roty
    base_header.user_float[2] = rotz

    base_header.user_float[3] = cmx
    base_header.user_float[4] = cmy
    base_header.user_float[5] = cmz

    # base_header.user_float = (rotx, roty, rotz, cmx, cmy, cmz)

    print("cmx ", base_header.user_float[3], "cmy ", base_header.user_float[4], "cmz ", base_header.user_float[5])
    print("------ BASE HEADER ------", base_header)

    ninstances = nslices * nreps
    # r = np.zeros((dims[0], dims[1], dims[2], dims[3]))
    r = data_array
    # print(data_array.shape)
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.image.Image.from_array(r, headers=base_header)
    connection.send(image_array)
    print(base_header)


def send_reconstructed_images(connection, data_array, acq_header):
    # the fonction creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

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

    dims = data_array.shape

    base_header = ismrmrd.ImageHeader()
    base_header.version = acq_header.version
    ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.channels = ncoils  # The coils have already been combined
    base_header.matrix_size = (data_array.shape[0], data_array.shape[1], data_array.shape[2])
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.image_index = 0
    base_header.image_series_index = 0
    base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    base_header.repetition = acq_header.repetition

    ninstances = nslices * nreps
    r = np.zeros((dims[0], dims[1], ninstances))
    r = data_array
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.image.Image.from_array(r, headers=base_header)
    connection.send(image_array)


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
    # print("eNx", eNx, "eNy", eNy, "eNz", eNz)

    # Initialise a storage array
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

    ninstances = nslices * nreps
    # print("Number of instances ", ninstances)

    im = np.zeros((eNx, eNy, nslices), dtype=np.complex64)
    # print("im shape ", im.shape)

    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"
    file = "/home/sns/fetal-brain-track/files/" + date_path + "-" + timestamp + "-centreofmass.txt"

    with open(file, 'w') as f:
        f.write('Centre-of-Mass Coordinates')

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # #  SETTING UP LOCALISER JUST ONCE # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #

    for acquisition in connection:
        # print(acquisition)
        imag = np.abs(acquisition.data.transpose(3, 2, 1, 0))
        # print("image dimensions ", imag.shape)

        ndim = imag.shape
        # print("ndim ", ndim)

        # Get crop image, flip and rotate to match with true Nifti image
        img = imag[:, :, :, 0]

        # Stuff into the buffer
        slice = acquisition.slice
        repetition = acquisition.repetition
        print("repetition idx ", repetition, "slice idx ", slice)

        im[:, :, slice] = np.squeeze(img[:, :, 0])

        # rotx = acquisition.user_float[0]
        # roty = acquisition.user_float[1]
        # rotz = acquisition.user_float[2]
        #
        # cmx = acquisition.user_float[3]
        # cmy = acquisition.user_float[4]
        # cmz = acquisition.user_float[5]

        if slice == nslices - 1:

            if nslices % 2 != 0:
                mid = int(nslices / 2) + 1
            else:
                mid = int(nslices / 2)
            print("This is the mid slice: ", mid)
            im_corr2a = im[:, :, 0:mid]
            im_corr2b = im[:, :, mid:]

            im_corr2ab = np.zeros(np.shape(im), dtype='complex_')

            im_corr2ab[:, :, ::2] = im_corr2a
            im_corr2ab[:, :, 1::2] = im_corr2b

            print("-------------------------------------------------------------------------")

            # logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

            # for s in range(nslices):
            #     plt.imshow(np.squeeze(np.abs(im[:, :, s, 0])), cmap="gray")
            #     plt.show()

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
                                                  root_dir='/home/sns',
                                                  csv_dir='/home/sns/fetal-brain-track/files/',
                                                  checkpoint_dir='/home/sns/fetal-brain-track/checkpoints/',
                                                  train_csv=
                                                  'data_localisation_1-label-brain_uterus_train.csv',
                                                  valid_csv=
                                                  'data_localisation_1-label-brain_uterus_valid.csv',
                                                  test_csv=
                                                  'data_localisation_1-label-brain_uterus_run.csv',
                                                  run_csv=
                                                  'data_localisation_1-label-brain_uterus_run.csv',
                                                  run_input=im_corr2ab,
                                                  results_dir='/home/sns/fetal-brain-track/results/',
                                                  exp_name='Loc_3D',
                                                  task_net='unet_3D',
                                                  n_classes=N_classes)

            # for acquisition in connection:

            args.gpu_ids = [0]

            # RUN with empty masks - to generate new ones (practical application)

            if args.running:
                print("Running")
                # print("im shape ", im_corr2ab.shape)

                model = md.LocalisationNetwork3DMultipleLabels(args)
                # Run inference
                ####################
                model.run(args, 1)  # Changing this to 0 avoids the plotting

                rotx = 0.0
                roty = 0.0
                rotz = 0.0

                xcm = model.x_cm
                ycm = model.y_cm
                zcm = model.z_cm

                CoM = xcm, ycm, zcm

                text = str('CoM: ')
                append_new_line(file, text)
                text = str(xcm)
                append_new_line(file, text)
                text = str(ycm)
                append_new_line(file, text)
                text = str(zcm)
                append_new_line(file, text)
                text = str('---------------------------------------------------')
                append_new_line(file, text)

                print("centre-of-mass coordinates: ", xcm, ycm, zcm)
                print("Localisation completed.")

                send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)

            x = xcm
            y = ycm
            z = zcm
            # del rotx, roty, rotz, xcm, ycm, zcm

        else:
            # send_reconstructed_images(connection, im_corr2ab, acquisition)
            if repetition == 0:
                rotx = 0.0
                roty = 0.0
                rotz = 0.0

                xcm = 0.0
                ycm = 0.0
                zcm = 0.0
                send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                # del rotx, roty, rotz, xcm, ycm, zcm

            else:
                rotx = 0.0
                roty = 0.0
                rotz = 0.0

                xcm = x
                ycm = y
                zcm = z
                send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                # del rotx, roty, rotz, xcm, ycm, zcm

            continue

# # # # # # # # # # # # # # # # # # # # # # #
