import numpy as np
import gadgetron
import ismrmrd
import logging
import time
import io
import os
from datetime import datetime

from ismrmrd.meta import Meta
import itertools
import ctypes
import numpy as np
import copy
import io
import warnings

warnings.simplefilter('default')

from ismrmrd.acquisition import Acquisition
from ismrmrd.flags import FlagsMixin
from ismrmrd.equality import EqualityMixin
from ismrmrd.constants import *

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

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def get_first_index_of_non_empty_header(header):
    # if the data is under-sampled, the corresponding acquisition Header will be filled with 0
    # in order to catch valuable information, we need to catch a non-empty header
    # using the following lines

    print(np.shape(header))
    dims = np.shape(header)
    for ii in range(0, dims[0]):
        # print(header[ii].scan_counter)
        if header[ii].scan_counter > 0:
            break
    print(ii)
    return ii


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
    ndims_image = (dims[0], dims[1], dims[2])

    base_header = ismrmrd.ImageHeader()
    base_header.version = 1
    # ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.measurement_uid = acq_header.measurement_uid
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.physiology_time_stamp = acq_header.physiology_time_stamp
    base_header.user_float[0] = rotx
    base_header.user_float[1] = roty
    base_header.user_float[2] = rotz
    base_header.user_float[3] = cmx
    base_header.user_float[4] = cmy
    base_header.user_float[5] = cmz

    # base_header.user_float = (rotx, roty, rotz, cmx, cmy, cmz)

    print("cmx ", base_header.user_float[3], "cmy ", base_header.user_float[4], "cmz ", base_header.user_float[5])
    # print("------ BASE HEADER ------", base_header)

    ninstances = nslices * nreps
    # r = np.zeros((dims[0], dims[1], dims[2], dims[3]))
    r = data_array
    # print(data_array.shape)
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.Image.from_array_wcm(rotx, roty, rotz, cmx, cmy, cmz, r, headers=acq_header)

    # image_array = ismrmrd.ImageHeader.from_acquisition(acq_header)
    print("..................................................................................")
    logging.info("Last slice of the repetition reconstructed - sending to the scanner...")
    connection.send(image_array)
    # print(base_header)
    logging.info("Sent!")
    print("..................................................................................")


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

    print("..................................................................................")
    logging.info("Sending reconstructed slice to the scanner...")
    connection.send(image_array)
    logging.info("Sent!")
    print("..................................................................................")


def IsmrmrdToNiftiGadget(connection):
    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    logging.info("Initializing data processing in Python...")
    # start = time.time()

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

    if enc.encodingLimits.contrast is not None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1
    print("Number of echoes =", ncontrasts)

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
    print("Image Shape ", im.shape)

    file = "/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "-" + timestamp + "-centreofmass.txt"
    # file = "/home/sns/fetal-brain-track/files/" + date_path + "-" + timestamp + "-centreofmass.txt"

    with open(file, 'w') as f:
        f.write('Centre-of-Mass Coordinates')

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # #  SETTING UP LOCALISER JUST ONCE # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #

    def motion(rx, ry, rz, tx, ty, tz):
        print("inside the motion function!")
        rx = rotx
        ry = roty
        rz = rotz
        tx = xcm
        ty = ycm
        tz = zcm

        print("motion parameters: ", rx, ry, rz, tx, ty, tz)

    com = []

    for acquisition in connection:
        # print(acquisition)
        imag = np.abs(acquisition.data.transpose(3, 2, 1, 0))
        print("Slice Dimensions ", imag.shape)

        ndim = imag.shape
        # print("ndim ", ndim)

        # Get crop image, flip and rotate to match with true Nifti image
        img = imag[:, :, :, 0]

        # Stuff into the buffer
        slice = acquisition.slice
        repetition = acquisition.repetition
        contrast = acquisition.contrast
        print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

        logging.info("Storing each slice into the 3D data buffer...")
        im[:, :, slice] = np.squeeze(img[:, :, 0])

        # rotx = acquisition.user_float[0]
        # roty = acquisition.user_float[1]
        # rotz = acquisition.user_float[2]
        #
        # cmx = acquisition.user_float[3]
        # cmy = acquisition.user_float[4]
        # cmz = acquisition.user_float[5]

        if ncontrasts == 1:  # only one echo-time
            # if the whole stack of slices has been acquired >> apply network to the entire 3D volume
            if slice == nslices - 1:
                logging.info("All slices stored into the data buffer!")
                # if nslices % 2 != 0:
                #     mid = int(nslices / 2) + 1
                # else:
                #     mid = int(nslices / 2)
                # print("This is the mid slice: ", mid)
                # im_corr2a = im[:, :, 0:mid]
                # im_corr2b = im[:, :, mid:]
                #
                # im_corr2ab = np.zeros(np.shape(im), dtype='complex_')
                #
                # im_corr2ab[:, :, ::2] = im_corr2a
                # im_corr2ab[:, :, 1::2] = im_corr2b

                print("..................................................................................")
                print("This is the echo-time we're looking at: ", contrast)
                # logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

                # for s in range(nslices):
                #     plt.imshow(np.squeeze(np.abs(im[:, :, s, 0])), cmap="gray")
                #     plt.show()

                # ==================================================================================================== #
                #
                #  TRAIN Localisation Network with 3D images
                #
                # ==================================================================================================== #
                logging.info("Initializing localization network...")

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
                                                      root_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python',
                                                      csv_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/files/',
                                                      checkpoint_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints/2022-12-16-newest/',
                                                      # change to -breech or -young if needed!
                                                      train_csv=
                                                      'data_localisation_1-label-brain_uterus_train-2022-11-23.csv',
                                                      valid_csv=
                                                      'data_localisation_1-label-brain_uterus_valid-2022-11-23.csv',
                                                      test_csv=
                                                      'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                      run_csv=
                                                      'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                      # run_input=im_corr2ab,
                                                      run_input=im,
                                                      results_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/',
                                                      exp_name='Loc_3D',
                                                      task_net='unet_3D',
                                                      n_classes=N_classes)

                # for acquisition in connection:

                args.gpu_ids = [0]

                # RUN with empty masks - to generate new ones (practical application)

                if args.running:
                    print("Running")
                    # print("im shape ", im_corr2ab.shape)
                    logging.info("Starting localization...")
                    model = md.LocalisationNetwork3DMultipleLabels(args)
                    # Run inference
                    ####################
                    model.run(args, 1)  # Changing this to 0 avoids the plotting
                    logging.info("Localization completed!")

                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    logging.info("Storing motion parameters into variables...")
                    xcm = model.x_cm
                    ycm = model.y_cm
                    zcm = model.z_cm
                    logging.info("Motion parameters stored!")

                    # # # # #

                    # CoM = xcm, ycm, zcm
                    # com_x = np.append(com, xcm)
                    # com_y = np.append(com, ycm)
                    # com_y = np.append(com, zcm)

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

                    if repetition == 0:
                        logging.info("Calculating translational motion in repetition 0...")
                        xfrep = xcm
                        yfrep = ycm
                        zfrep = zcm

                        tx_prev2 = 0.0
                        ty_prev2 = 0.0
                        tz_prev2 = 0.0

                        tx_prev = 0.0
                        ty_prev = 0.0
                        tz_prev = 0.0

                        x = 0.0
                        y = 0.0
                        z = 0.0

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # cm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        tx = 0.0
                        ty = 0.0
                        tz = 0.0

                        tx_prev = 0.0
                        ty_prev = 0.0
                        tz_prev = 0.0

                        # tx2 = 0.0
                        # ty2 = 0.0
                        # tz2 = 0.0

                        logging.info("Motion calculated!")
                        print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

                    else:
                        print("xfrep yfrep zfrep", xfrep, yfrep, zfrep)
                        logging.info("Calculating motion parameters in current repetition...")
                        # tx = (xcm - xcm_prev) * 3.0 + tx_prev
                        # ty = (ycm - ycm_prev) * 3.0 + ty_prev
                        # tz = (zcm - zcm_prev) * 3.0 + tz_prev

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = xcm_prev

                        # xcm_prev = xcm
                        # ycm_prev = ycm
                        # zcm_prev = zcm

                        tx = (xcm - xcm_prev) * 3.0 + tx_prev2
                        ty = (ycm - ycm_prev) * 3.0 + ty_prev2
                        tz = (zcm - zcm_prev) * 3.0 + tz_prev2

                        print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                        print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                        print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                        xcm_prev2 = xcm_prev
                        ycm_prev2 = ycm_prev
                        zcm_prev2 = xcm_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # tx2 = (xcm - xcm_prev2) * 3.0
                        # ty2 = (ycm - ycm_prev2) * 3.0
                        # tz2 = (zcm - zcm_prev2) * 3.0

                        # print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                        # print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                        # print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                        # tx = 0.0
                        # ty = 0.0
                        # tz = 0.0

                        print("I am slice ", slice, " and I am applying shift ", tx, ty, tz)
                        print("CoM coordinates for the repetition: ", xcm, ycm, zcm)
                        print("Translational motion for the repetition: ", tx, ty, tz)

                        tx_prev2 = tx_prev
                        ty_prev2 = ty_prev
                        tz_prev2 = tz_prev

                        tx_prev = tx
                        ty_prev = ty
                        tz_prev = tz

                        # tx_prev2 = tx_prev
                        # ty_prev2 = ty_prev
                        # tz_prev2 = tz_prev

                        xcm_prev = xcm
                        ycm_prev = ycm
                        zcm_prev = zcm

                        # xcm_prev2 = xcm_prev
                        # ycm_prev2 = ycm_prev
                        # zcm_prev2 = zcm_prev

                        x = -ty  # -ty
                        y = tz  # tz
                        z = tx  # tx

                        # IF THERE IS A DELAY OF TWO REPETITIONS
                        # x = -ty2
                        # y = tx2
                        # z = tz2

                        logging.info("Motion calculated!")
                        print("here's x y and z", x, y, z)

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, x, y, z, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)

                # x = xcm
                # y = ycm
                # z = zcm

                x = -ty  # -ty # seems to work
                y = tz  # tz
                z = tx  # tx  # seems to work

                # if repetition == 0:
                #     xfrep = xcm
                #     yfrep = ycm
                #     zfrep = zcm
                #
                #     print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

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
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                else:
                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    tx_ = x
                    ty_ = y
                    tz_ = z

                    # print("I am slice ", slice, " and I am applying shift ", tx_, ty_, tz_)

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, tx_, ty_, tz_, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                continue

            # return rotx, roty, rotz, xcm, ycm, zcm

        else:  # for multi-echo acquisitions!
            if contrast == 0:
                # if the whole stack of slices has been acquired >> apply network to the entire 3D volume
                if slice == nslices - 1:  # if last slice & second echo-time
                    logging.info("All slices stored into the data buffer!")
                    # if nslices % 2 != 0:
                    #     mid = int(nslices / 2) + 1
                    # else:
                    #     mid = int(nslices / 2)
                    # print("This is the mid slice: ", mid)
                    # im_corr2a = im[:, :, 0:mid]
                    # im_corr2b = im[:, :, mid:]
                    #
                    # im_corr2ab = np.zeros(np.shape(im), dtype='complex_')
                    #
                    # im_corr2ab[:, :, ::2] = im_corr2a
                    # im_corr2ab[:, :, 1::2] = im_corr2b

                    # logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

                    # for s in range(nslices):
                    #     plt.imshow(np.squeeze(np.abs(im[:, :, s, 0])), cmap="gray")
                    #     plt.show()

                    # ==================================================================================================== #
                    #
                    #  TRAIN Localisation Network with 3D images
                    #
                    # ==================================================================================================== #

                    print("..................................................................................")
                    print("This is the echo-time we're looking at: ", contrast)

                    logging.info("Initializing localization network...")

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
                                                          root_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python',
                                                          csv_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/files/',
                                                          checkpoint_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints/2022-12-16-newest/',
                                                          # change to -breech or -young if needed!
                                                          train_csv=
                                                          'data_localisation_1-label-brain_uterus_train-2022-11-23.csv',
                                                          valid_csv=
                                                          'data_localisation_1-label-brain_uterus_valid-2022-11-23.csv',
                                                          test_csv=
                                                          'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                          run_csv=
                                                          'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                                          # run_input=im_corr2ab,
                                                          run_input=im,
                                                          results_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/',
                                                          exp_name='Loc_3D',
                                                          task_net='unet_3D',
                                                          n_classes=N_classes)

                    # for acquisition in connection:

                    args.gpu_ids = [0]

                    # RUN with empty masks - to generate new ones (practical application)

                    if args.running:
                        print("Running")
                        # print("im shape ", im_corr2ab.shape)
                        logging.info("Starting localization...")
                        model = md.LocalisationNetwork3DMultipleLabels(args)
                        # Run inference
                        ####################
                        model.run(args, 1)  # Changing this to 0 avoids the plotting
                        logging.info("Localization completed!")

                        rotx = 0.0
                        roty = 0.0
                        rotz = 0.0

                        logging.info("Storing motion parameters into variables...")
                        xcm = model.x_cm
                        ycm = model.y_cm
                        zcm = model.z_cm
                        logging.info("Motion parameters stored!")

                        # # # # #

                        # CoM = xcm, ycm, zcm
                        # com_x = np.append(com, xcm)
                        # com_y = np.append(com, ycm)
                        # com_y = np.append(com, zcm)

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

                        if repetition == 0:
                            logging.info("Calculating translational motion in repetition 0...")
                            xfrep = xcm
                            yfrep = ycm
                            zfrep = zcm

                            tx_prev2 = 0.0
                            ty_prev2 = 0.0
                            tz_prev2 = 0.0

                            tx_prev = 0.0
                            ty_prev = 0.0
                            tz_prev = 0.0

                            x = 0.0
                            y = 0.0
                            z = 0.0

                            # xcm_prev2 = xcm_prev
                            # ycm_prev2 = ycm_prev
                            # zcm_prev2 = xcm_prev

                            xcm_prev = xcm
                            ycm_prev = ycm
                            zcm_prev = zcm

                            # IF THERE IS A DELAY OF TWO REPETITIONS
                            # cm_prev2 = xcm_prev
                            # ycm_prev2 = ycm_prev
                            # zcm_prev2 = xcm_prev

                            tx = 0.0
                            ty = 0.0
                            tz = 0.0

                            tx_prev = 0.0
                            ty_prev = 0.0
                            tz_prev = 0.0

                            # tx2 = 0.0
                            # ty2 = 0.0
                            # tz2 = 0.0

                            logging.info("Motion calculated!")
                            print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

                        else:
                            logging.info("Calculating motion parameters in current repetition...")
                            print("xfrep yfrep zfrep", xfrep, yfrep, zfrep)
                            # tx = (xcm - xcm_prev) * 3.0 + tx_prev
                            # ty = (ycm - ycm_prev) * 3.0 + ty_prev
                            # tz = (zcm - zcm_prev) * 3.0 + tz_prev

                            # xcm_prev2 = xcm_prev
                            # ycm_prev2 = ycm_prev
                            # zcm_prev2 = xcm_prev

                            # xcm_prev = xcm
                            # ycm_prev = ycm
                            # zcm_prev = zcm

                            tx = (xcm - xcm_prev) * 3.0 + tx_prev2
                            ty = (ycm - ycm_prev) * 3.0 + ty_prev2
                            tz = (zcm - zcm_prev) * 3.0 + tz_prev2

                            print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                            print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                            print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                            xcm_prev2 = xcm_prev
                            ycm_prev2 = ycm_prev
                            zcm_prev2 = xcm_prev

                            xcm_prev = xcm
                            ycm_prev = ycm
                            zcm_prev = zcm

                            # IF THERE IS A DELAY OF TWO REPETITIONS
                            # tx2 = (xcm - xcm_prev2) * 3.0
                            # ty2 = (ycm - ycm_prev2) * 3.0
                            # tz2 = (zcm - zcm_prev2) * 3.0

                            # print("xcm xcm_prev tx_prev tx_prev2 tx: ", xcm, xcm_prev, tx_prev, tx_prev2, tx)
                            # print("ycm ycm_prev ty_prev ty_prev2 ty: ", ycm, ycm_prev, ty_prev, ty_prev2, ty)
                            # print("zcm zcm_prev tz_prev tz_prev2 tz: ", zcm, zcm_prev, tz_prev, tz_prev2, tz)

                            # tx = 0.0
                            # ty = 0.0
                            # tz = 0.0

                            print("I am slice ", slice, " and I am applying shift ", tx, ty, tz)
                            print("CoM coordinates for the repetition: ", xcm, ycm, zcm)
                            print("Translational motion for the repetition: ", tx, ty, tz)

                            tx_prev2 = tx_prev
                            ty_prev2 = ty_prev
                            tz_prev2 = tz_prev

                            tx_prev = tx
                            ty_prev = ty
                            tz_prev = tz

                            # tx_prev2 = tx_prev
                            # ty_prev2 = ty_prev
                            # tz_prev2 = tz_prev

                            xcm_prev = xcm
                            ycm_prev = ycm
                            zcm_prev = zcm

                            # xcm_prev2 = xcm_prev
                            # ycm_prev2 = ycm_prev
                            # zcm_prev2 = zcm_prev

                            x = -ty  # -ty
                            y = tz  # tz
                            z = tx  # tx

                            # IF THERE IS A DELAY OF TWO REPETITIONS
                            # x = -ty2
                            # y = tx2
                            # z = tz2

                            logging.info("Motion calculated!")
                            print("here's x y and z", x, y, z)

                        send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, x, y, z, acquisition)
                        # motion(rotx, roty, rotz, xcm, ycm, zcm)
                        # send_reconstructed_images(connection, imag, acquisition)

                    # x = xcm
                    # y = ycm
                    # z = zcm

                    # calculated motion parameters to be passed onto all slices but last of the next repetition
                    x = -ty  # -ty # seems to work
                    y = tz  # tz
                    z = tx  # tx  # seems to work

                    # if repetition == 0:
                    #     xfrep = xcm
                    #     yfrep = ycm
                    #     zfrep = zcm
                    #
                    #     print("These are CoM coordinates for first repetition: ", xfrep, yfrep, zfrep)

                else:  # if it's not the last slice of the repetition (still in TE2!)
                    # send_reconstructed_images(connection, im_corr2ab, acquisition)
                    if repetition == 0:
                        rotx = 0.0
                        roty = 0.0
                        rotz = 0.0

                        xcm = 0.0
                        ycm = 0.0
                        zcm = 0.0

                        send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                        # motion(rotx, roty, rotz, xcm, ycm, zcm)
                        # send_reconstructed_images(connection, imag, acquisition)
                        # del rotx, roty, rotz, xcm, ycm, zcm

                    else:
                        rotx = 0.0
                        roty = 0.0
                        rotz = 0.0

                        tx_ = x
                        ty_ = y
                        tz_ = z

                        # print("I am slice ", slice, " and I am applying shift ", tx_, ty_, tz_)

                        send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, tx_, ty_, tz_, acquisition)
                        # motion(rotx, roty, rotz, xcm, ycm, zcm)
                        # send_reconstructed_images(connection, imag, acquisition)
                        # del rotx, roty, rotz, xcm, ycm, zcm

                    continue

                # calculated motion parameters to be passed onto all slices but last of the next repetition
                x = -ty  # -ty # seems to work
                y = tz  # tz
                z = tx  # tx  # seems to work

            else:  # for all other echo-times but TE2!
                if repetition == 0:  # for the first repetition of all TEs but TE2
                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    xcm = 0.0
                    ycm = 0.0
                    zcm = 0.0

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                else:  # all repetitions but first for all echo-times but TE2
                    rotx = 0.0
                    roty = 0.0
                    rotz = 0.0

                    tx_ = x
                    ty_ = y
                    tz_ = z

                    # print("I am slice ", slice, " and I am applying shift ", tx_, ty_, tz_)

                    send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, tx_, ty_, tz_, acquisition)
                    # motion(rotx, roty, rotz, xcm, ycm, zcm)
                    # send_reconstructed_images(connection, imag, acquisition)
                    # del rotx, roty, rotz, xcm, ycm, zcm

                continue

# # # # # # # # # # # # # # # # # # # # # # # # #
