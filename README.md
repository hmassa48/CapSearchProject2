# Tunable U-Net Architecture

This project works to answer the question:  Is there a best U-Net architecture? 

It does this through the use of the keras-tuner. It is important to run the keras-tuner with the correct form of tensorflow and keras that you will need. This code is optimized through the use of GPUs, so it will need to be run with tensorflow-gpu. To run these packages, you must install the keras-tuner and then uninstall tensorflow and reinstall the tensorflow-GPU making sure that the keras-tuner doesn't get deleted in the process. 

To run the tuner, it is set up to run similar to the in keras models. 

## Running the Tunable U-Net 

### Full U-Net Search

This work created a tunable U-Net architecture using the Keras-Tuner. To run the tunable U-Net, you have to import the HyperUNet class from the keras models folder and state the input shape as well as the number of classes. 

hypermodel = HyperUNet(input_shape, num_classes) 

This will search all possible U-Net parameters listed. This work focused on the architectural aspects of the U-Net, that can be found and more explained in the section below. 

### Architectural U-Net Search for Project

For this work, I focused on the architectural aspects of the U-Net model. For this reason, I fixed some hyperparameters in the tunable U-Net so that only the 8 parameters I wanted to tune were tuned. This can be done and is shown in the image below. 

![Fixed Parameters](/Images/FixedParameters.png)


This image shows the values needed to input to the model. Then, it also shows the structuring of the overall tuner along with how to initiate a search. The images will have to be pre-processed between importing the necessary packages and using the search strategy. This provides a quick introduction on how to use the HyperUNet class, but full search examples are shown in python scripts such as tuneChest.py and tuneSkin.py. 

### Other Tunable U-Nets 

Work is being done to add more tunable U-Nets to the repository. 

## Training for Data Sets included in Project 

This project ran the tunable U-Net network to evaluate the best architecture for four different datasets. These datasets can be found listed below, but there were different pre-processing and training techniques used for each. A toy dataset for each of the models has been added. 

![Data Set Examples](/Images/DataSetImage.png)

### Lung Segmentation U-Net

There are two scripts to run for each of the dataset values. The first script is the trainLungUnet.py. This script runs the U-Net model. Within this script, the hyperparameters can be changed to match the traditional U-Net, or a tuned version of the U-Net. 

To run this model on your own system use: python trainLungUNet.py
To run this model on Euler use: sbatch slurm_Lung_basic_Unet.sh

Then to tune the model, the associating tuning script is the tuneChest.py script. This script utilizes the HyperUNet class described above. This script will output a textfile with the chosen hyperparameters of the best 10 models found from the HyperUNet search. This allows the user to compare the top models found. 

To run this model on your own system use: python tuneChest.py
To run this model on Euler use: sbatch slurm_chestTune.sh

After the top models have been found, this work took the parameters from the best architecture found from the HyperUNet and used them in the script that runs the U-Net model. In other words, I changed the parameters within the custom U-Net of trainLungUNet.py script and re-ran the trainLungUNet.py script. 

### LGG Brain MR Segmentation U-Net

This dataset followed the same techniques as above. Therefore, I am just going to explain how to run the scripts. 

To run the traditional U-Net, or tuned version after changing the parameters within the script: 
To run this model on your own system use: python trainBasicUNet.py
To run this model on Euler use: sbtach slurm_LGG_basicUnet.sh

To run the Tuning script:
To run this model on your own system use: python tune
To run this model on Euler use: sbtach slurm_onetune.sh

### Skin Lesion Segmentation U-Net

This dataset followed the same techniques as above. Therefore, I am just going to explain how to run the scripts. 

To run the traditional U-Net, or tuned version after changing the parameters within the script: 
To run this model on your own system use: python trainSkinUNet.py
To run this model on Euler use: sbtach slurm_Skin_basicUNet.sh

To run the Tuning script:
To run this model on your own system use: python tune
To run this model on Euler use: sbatch slurm_skinTune.sh

## Evaluating the Models 

Jupyter notebooks were created for each of the datasets to be evaluated. 


## Running on the Euler Cluster 

Run the environment saved in the files above

module load anaconda/wml
bootstrap_conda

conda create --name env697 --file env697.txt

conda activate env697


Submit the Slurm directories saved in the slurm folder above to submit jobs 
