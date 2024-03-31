CNN Model for Brain Tumor Detection 🧠
-----------------------------------
This is a great project to get used to the PyTorch fundamentals

The folder includes:
- .csv file for ```__getitem__``` in the dataset object (```csv_labels.py```)
- Generated .csv file with image name and targets
- Custom dataset code (```data.py```)
- CNN Model (```model.py```)
- Train and test code (```train.py```)

⬇️ Dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)!

Notes
-----------------------------------
The code ```data.py``` includes a static method to identify the std and mean of the image dataset. These are later used to normalize the data in the transformations.

The ```train.py``` includes an unusual method for the training loop. First, it includes a function ```train_one_epoch()``` that incluides the backbone of the PyTorch training loop. 
Then, it uses that function within the ```train()``` function, which takes epochs as an argument.

In the best run, I got an accuracy of 100% on the training set and 87% on the test set. However, this could be improved if you implemented data augmentation.
