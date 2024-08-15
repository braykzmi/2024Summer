# Brayden Mi 2024 Summer Work
## Usage of Setup Scripts
I've included 2 different setup scripts for biliary duct/liver training on nn-Unet and MIST respectively. These scripts take in raw data in dicom format, converts it into Niftis and organizes it into a data format that will be accepted by either nn-Unet or MIST. **This does not include writing the dataset.json file for either nn-Unet or MIST**

**Dependencies**

Make sure to have [Convert3d](https://sourceforge.net/projects/c3d/files/), Python 3.X, and [dcmrtstruct2nii](https://github.com/Sikerdebaard/dcmrtstruct2nii) installed. 

**1. Editing the Script**
For either, please first edit the scripts to reflect the path to the parent directory of the data. You can do this via the nano command with:

`nano setupunet.sh`

At the top, change the string for PARENT_DIR.

**2. Setting up the dataset directory**
In the directory, make a raw_trainingdata directory. This contain all the dicom training data. Please be sure that it is organized in this format:

```
raw_trainingdata
└──Some MRN
  └──Study Directory
    └──DICOM Image Directory
      └──DICOM Images
    └──RTStruct Directory
      └──RTStruct File
└──Some other MRN
  └──Same as above
...
```

Make the following directories in the parent folder: biliary_labels, liver_labels, tumor_labels, segmentations, cases, tmp. If setup for MIST, make a mist_data directory as well. Finally, make a blank csv file called mrn_casenum.csv, which will contain the MRN and case number table.

**3. Running the Script**
Run the script with the bash command. ex:

`bash setupmist.sh`

***Error notes***

Since these scripts run based on DICOM files and use both Convert3d and dcmrtstruct2ii, there are some cases where there will be a dimension mismatch error while converting and performing image operations to combine segmentations. The script will list this in the command line with the message "$MRN_NUM error" and skip the case.

**4. Moving the Data**

For MIST, moving is not necessary since the data path can be specified in the dataset.json file for MIST. If needed, simply move the data from the mist_data directory to the MIST working environment data directory. More information on MIST can be found [here](https://mist-medical.readthedocs.io/en/latest/).

For nnUNet, move the data from the cases directory into the imagesTr directory for the nnUNet dataset and then move the data from the segmentations directory into the labelsTr directory. More information on the nnUnet setup can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).


## Usage of Python Programs
### 1. TensorVoting.py
This program generates reconnection segments for a segmentation via Tensor Voting. It takes in 3 files: Segmentation file, Skeleton file, and Radius file. Making these files is as follows:

**Segmentation File**: Should already have this generated from somewhere

**Skeleton File**: Use ITKThinningFilter to thin the segmentation file from before

**Radius file**: Conduct a signed distance transform on the segmentation and multiply with the skeleton file as follows:
`c3d <segmentation file>.nii.gz -sdt <skeleton file>.nii.gz -multiply -o <radius file>.nii.gz`

Once ran through the program, add the outputted file on top of the original segmentation to include the reconnections.

### 2. SoftSkeleton2.py
This program does the softskeletonize algorithm from the clDice paper. To use, edit the file and change the input path to the filepath of the segmentation to be softskeletonized. Do not use SoftSkeleton.py.

### 3. distance_transform.py
Incomplete program. Will not work/do not use.
