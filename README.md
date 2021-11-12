# affectnet-preprocessing-training
Notebook that processes the AffectNet dataset and trains a CNN on the images. This project was completed with the goal to create a Sagemaker pipeline that would automate this process.

## Dependencies
- os: Join strings into a filepath.
- cv2: Read in images and resize them.
- numpy: Normalize pixel values for each face and store train/test/validation datasets.
- pandas: Read in the csv file as a dataframe and manipulate it. Use another dataframe to house the processed faces and corresponding expressions.
- tensorflow: Build, compile, and train a CNN.

## Dataset
This project works with the [AffectNet dataset](http://mohammadmahoor.com/affectnet/). The dataset has over 400,000 pictures of faces that belong to 11 categories. For this particular project, the five most numerous expressions were used (neutral, happy, sad, surprise, and anger). The dataset comes with many csv files and two folders of images. This project uses the "automatically_annotated.csv" file to locate and process images found in the "Automatically_Annotated" folder. Note that filepaths in my code are based on an AWS EC2 instance. You may need to change the filepath variables to get this to work for you.

## Processing
The automatically_annotated csv is read into the program and unnecessary columns are dropped from the dataframe. The unwanted expressions are dropped from the table, and the index is reset. An empty dataframe for the processed data is created, and a loop runs over the entire automatically_annotated dataframe to extract the faces from the images. Each image is read into the program in grayscale, from which the face is extracted, resized, and normalized. The face is saved with the corresponding expression to the new dataframe. A counter tracks progress during this step (it took about 2 hours to complete on the EC2 instance).

The dataframe with the faces and expressions is split by expression. The dataframes for each expression are shortened to the same length to balance the dataset. Index slicing is used to split the data into train/test/validation subsets. The faces and expressions are stored in numpy arrays for model training. Interesting note: the CNN model with 5 categories would not accept a label greater than 5, so anger had to be remapped from expression 6 to 4 with a simple helper function.

## Results
A basic CNN is defined and compiled using tensorflow. Please note: the model seen here is a simplified version of the one I actually used. I simplified it before posting to protect the privacy of the model used by Emsyte.

![image](https://user-images.githubusercontent.com/83524079/141444811-c1f2ff14-9a2b-40e6-977a-22c7c3c893fa.png)

The accuracy improved greatly with each epoch. For the first attempt to train the model, I was pleasantly surprised by the results.

## Future work
The main future work to be done here is split the notebook into data preprocessing and model training scripts for use in a Sagemaker Pipeline. Other future work could be done to improve the accuracy of the model by tuning the hyperparameters and introducing some data augmentation.
