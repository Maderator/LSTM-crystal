# Lost Access to Matlab License
- We have unfortunately lost access to the Metacentrum and Matlab license due to
interruption of studies and subsequent loss of student status. 
- We therefore cannot check the correctness of the Matlab code and the correctness of Matlab experiments.
- Therefore, treet code in Matlab folder as a draft and not as a final version.

# Data Preparation

## How to obtain dataset
Feel free to contact me at `janmadera97+mff_thesis_data at gmail.com` and I will ask for premission to share the dataset with you.

If the email address is not available, please, try it again without the `+mff_thesis_data` part.

## Split dataset
1. Save the dataset to the `data/sequencesInputsOutputs.mat` file.
2. Run the `dataManipulation/splitFinalTestingData.m` script to prepare data splits for training and testing.
3. Now you need only to create Matlab project and add the whole (matlab) repository to the project and you can start testing the Matlab custom LSTM layers implementations.

Keep in mind that we have lost access to the Matlab license and therefore we could not check the correctness of the Matlab code and the correctness of Matlab experiments. Therefore, treat code in Matlab folder as a draft and not as a final version.

Python code is the fully functional, tested, and fast version with all the LSTM layers and cells implemented. 