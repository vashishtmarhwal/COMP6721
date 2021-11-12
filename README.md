###############################################
##                                           ##
## COMP 6721 Applied Artificial Intelligence ##
##                  Phase I                  ##
##                                           ##
###############################################

##

1. File Struxture
------------------

    a. main.py - Contains the model, the training and data pre-processing code
    b. FinalDataset - Contains the dataset of images belonging to cloth, ffp2, surgical, without_mask category
    c. k_cross-CNN.pt - Pre-Trained CNN Model
    d. testImgage - Contains never before seen images by the model for testing
    e. ConfusionMatrix.png - Confusion Matrix generated at fold-4 while training
    f. DatasetSat.png - Bargraph showing the number of images in each class
    g. 6721 Report.pdf - Project Report containing the required info on Dataset, CNN Model and Evaluation

2. How to train a new CNN Model
--------------------------------

    a. Delete the k_cross_CNN.pt file
    b. run python main.py

3. Test pre-trained model on new images outside dataset
--------------------------------------------------------

    a. run python main.py which will load k_cross_CNN.pt
    b. A random class label between 0 and 3 will be picked on which prediction will be made. 

