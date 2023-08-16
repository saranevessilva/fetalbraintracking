# Fetal Brain Localization with Gadgetron

### Step 1:
Convert the raw Siemens data to ISMRMRD format using siemens_to_ismrmrd:
```bash
siemens_to_ismrmrd -f meas_MID00116_FID30450_ep2d_fid_ME_cor_p2_e5_wholeuterus.dat --skipSyncData -x IsmrmrdParameterMap_Siemens_NX_SNS.xsl -z 2 -o meas_MID00116_FID30450_ep2d_fid_ME_cor_p2_e5_wholeuterus.h5
```
Copy the parameter map IsmrmrdParameterMap_Siemens_NX_SNS.xml (modified Siemens software version NX50) to /gadgetron/dep-build/siemens_to_ismrmrd/parameter_maps/. Use the parameter map provided for converting the dataset. If the new parameter map is not recognised, copy the changes into the IsmrmrdParameterMap_Siemens_NX.xml.

### Step 2: 
Copy the converted dataset (.h5) to /miniconda3/envs/gadgetron/share/gadgetron/config.

### Step 3: 
Copy the provided configuration epi_external_python_localiser.xml file /miniconda3/envs/gadgetron/share/gadgetron/config.

### Step 4: 
Copy the following folders into /miniconda3/envs/gadgetron/share/gadgetron/python: src, checkpoints, files, results.

### Step 5: 
Copy the Python program into /miniconda3/envs/gadgetron/share/gadgetron/python, e.g., nifti_python_gadgetron_multi-echo.py.

### Step 6: 
In the Python program, make sure the paths to the above mentioned directories are correct, /home/sn21//miniconda3/envs/gadgetron/share/gadgetron/checkpoints.

### Step 7: 
You are good to go!

```bash
gadgetron_ismrmrd_client -f meas_MID00116_FID30450_ep2d_fid_ME_cor_p2_e5_wholeuterus.h5 -C epi_external_python_localiser.xml -o meas_MID00116_FID30450_ep2d_fid_ME_cor_p2_e5_wholeuterus_r001.h5
```
