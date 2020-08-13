# Tunable U-Net Architecture

This project works to answer the question:  Is there a best U-Net architecture? 

It does this through the use of the keras-tuner. It is important to run the keras-tuner with the correct form of tensorflow and keras that you will need. This code is optimized through the use of GPUs, so it will need to be run with tensorflow-gpu. To run these packages, you must install the keras-tuner and then uninstall tensorflow and reinstall the tensorflow-GPU making sure that the keras-tuner doesn't get deleted in the process. 

To run the tuner, it is set up to run similar to the in keras models. 

## Running the Tunable U-Net 

### Full U-Net Search

This work created a tunable U-Net architecture using the Keras-Tuner. To run the tunable U-Net, you have to import the HyperUNet class from the keras models folder and state the input shape as well as the number of classes. 



### Architectural U-Net Search for Project

For this work, I focused on the architectural aspects of the U-Net model. For this reason, I fixed some hyperparameters in the tunable U-Net so that only the 8 parameters I wanted to tune were tuned. This can be done and is shown in the image below. 

![Fixed Parameters](/Images/FixedParameters.png)

### Other Tunable U-Nets 

Work is being done to add more tunable U-Nets to the repository. 

## Training for Data Sets included in Project 

This project ran the tunable U-Net network to evaluate the best architecture for four different datasets. These datasets can be found listed below, but there were different pre-processing and training techniques used for each. A toy dataset for each of the models has been added. 

![Data Set Examples](/Images/DataSetImage.png)

### Lung Segmentation U-Net

To run the Lung Dataset, you need to run the python script for the 

### LGG Brain MR Segmentation U-Net

### Skin Lesion Segmentation U-Net

## Running on the Euler Cluster 

Run the environment saved in the files above

module load anaconda/wml
bootstrap_conda

conda activate env697

Submit the Slurm directories saved in the slurm folder above to submit jobs 
