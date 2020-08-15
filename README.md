# Tunable U-Net Architecture

This project works to answer the question:  Is there a best U-Net architecture? 

It does this through the use of the keras-tuner. It is important to run the keras-tuner with the correct form of tensorflow and keras that you will need. This code is optimized through the use of GPUs, so it will need to be run with tensorflow-gpu. To run these packages, you must install the keras-tuner and then uninstall tensorflow and reinstall the tensorflow-GPU making sure that the keras-tuner doesn't get deleted in the process. 

To run the tuner, it is set up to run similar to the in keras models. 

## Running the Tunable U-Net 

### Basic U-Net Search

This work created a tunable U-Net architecture using the Keras-Tuner. To run the tunable U-Net, you have to import the HyperBasicUNet class from the keras models folder and state the input shape as well as the number of classes. The HyperBasicUNet class defined for this work only tunes 8 architectural parameters, more parameters can be added within the HyperBasicUNet class. 

![Fixed Parameters](/Images/hyperunet.png)


This image shows the values needed to input to the model. Then, it also shows the structuring of the overall tuner along with how to initiate a search. The images will have to be pre-processed between importing the necessary packages and using the search strategy. This provides a quick introduction on how to use the HyperUNet class, but full search examples are shown in python scripts, explained more below. 

### Other Tunable U-Nets 

Work is being done to add more tunable U-Nets to the repository. 

## Training for Data Sets included in Project 

This project ran the tunable U-Net network to evaluate the best architecture for four different datasets. These datasets can be found listed below, but there were different pre-processing and training techniques used for each. A toy dataset for each of the models has been added to the repository to see how the images look. A toy dataset for the Lung images has been added to run and evaluate that code. 

![Data Set Examples](/Images/DataSetImage.png)

### Set up Environment for training 

There are many packages required to run this program. Set up for the program is done with an anaconda environment. To set up the environment, you have to set up and activate the enviroment with the following lines: 

conda create --name env697 --file env697.txt

conda activate env697

pip install tensorflow-gpu (dependency, has to be installed afterwards)

* the slurm files for training have been moved into a separate folder for visual sanity. If you want to run on Euler, the slurm files have to be added to the main directory. 

### Lung Segmentation U-Net

There are two scripts to run for each of the dataset values. The first script is the trainLungUnet.py. This script runs the U-Net model. Within this script, the hyperparameters can be changed to match the traditional U-Net, or a tuned version of the U-Net. 

To run this model on your own system use: python trainLungUNet.py
To run this model on Euler use: sbatch slurm_Lung_basic_Unet.sh

Then to tune the model, the associating tuning script is the tuneChest.py script. This script utilizes the HyperUNet class described above. This script will output a textfile with the chosen hyperparameters of the best 10 models found from the HyperUNet search. This allows the user to compare the top models found. 

To run this model on your own system use: python tuneChest.py
To run this model on Euler use: sbatch slurm_chestTune.sh

After the top models have been found, this work took the parameters from the best architecture found from the HyperUNet and used them in the script that runs the U-Net model. In other words, I changed the parameters within the custom U-Net of trainLungUNet.py script and re-ran the trainLungUNet.py script.

** Note for Capstone ** The Lung Dataset is the dataset with enough images able to be loaded into GitHub for a toy dataset for a full training and tuning. Due to GPU dependencies, the Hyper Class throws an eager tensor error when the dataset is too large and being trained on a GPU for an unspecified number of epochs. For this, I have added in a solution I used in the project when this came up. I had to build in the class into the actual run file and not collect anything from elsewhere. It is an odd dependency error others online have mentioned. Since it happens randomly and the keras-tuner team has put up a post saying they are working on fixing it, I have set up the main code as the HyperBasicUNet class. But due to the fact that it might break during tuning, I have added the chestTune2.py which builds it within the class with enough parameters for it to be able to run safely always. The keras-tuner is still a bit finnicky so it will break and need to restart. It is another reason why I computed the final models through changing the customizable U-Net with information given by the keras-tuner. It shows promising results consistently, but in this first version of the package it still has a lot of bugs. 

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

Jupyter notebooks were created for each of the datasets to be evaluated. The evaluation metrics were performed on saved models. The model files were too large to upload into github, so they do not have a toy dataset. 



