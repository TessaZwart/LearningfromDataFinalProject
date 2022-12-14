# LearningfromDataFinalProject ๐
## Explanation of repository

Classification of offensive tweets by using a classic model, LSTM, and language model. Additionally, we investigated if data augmentation using English-Spanish back translation, can improve the language model. Our research paper can be found here:

* Masoud Repazour Najafi, Nils Visser, and Tessa Zwart, 2022. โWhat the MIERDA!โ Classifying offensive tweets in combination with data augmentation techniques (English-Spanish back translation)

## Chapter 1: Installation Instructions
### 0. Cloning the repository
Install git and clone the repository. Use the following command to clone the repository:
```
git clone https://github.com/TessaZwart/LearningfromDataFinalProject.git
```

### 1. Creating virtual environment
Create a virtual environment by using the following line:
```
source env/bin/activate
```

### 2. Installing Dependencies
Make sure to install all dependencies before the program is executed by using python's package manager. Use the following command:
```
pip install -r requirements.txt
```


### 3. Downloading Spacy's language model
Make sure download and install spacy's language model by using the command:
```
python3 -m spacy download en_core_web_sm
```

### 4. Preprocess the data
Run the following command line to preprocess the data:
```
python3 preprocessing.py
```


## Chapter 2: Users manual
The hyperparameters that are chosen are tuned on the data of the following paper:
* Predicting the Type and Target of Offensive Posts in Social Media. Zampieri et. al
(2019)


### 0. Using classic model
To use the classic models, you can use the following command:
```
python3 baseline.py
```
The following models are implemented in this file: Logistic regression, Naive Bayes, Decision Tree, Random Forest, K-Neurest Neighbor, SVM, and Linear SVM.

### 1. Using Enhanced classic model (Naive Bayes)
To use the enhanced Naive Bayes model, you can use the following command:
```
python3 baseline_enhanced.py
```

### 2. Using LSTM model
To use the LSTM model, you first have to load the Glove embeddings. This can be done using the following commands:
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove*.zip
```
Then, you can run the model using the following command:
```
python3 LSTM.py
```

### 3. Using Language model
Unfortunately due to time contraints we were not able to implement the language model into our repository. a .ipynb file can be loaded into a free version of google colab (https://colab.research.google.com/) to run this file. Make sure to load your data first before you run the code.


## Chapter 3: How to train the model on own unseen data set
Different data can be used by changing the input data, using --train_file <OWN_TRAIN_FILE>, --dev_file <OWN_DEV_FILE>, or --test_file <OWN_TEST_FILE>, after the comment line. For example, running the LSTM model:
```
python3 LSTM.py --train_file <OWN_TRAIN_FILE> --dev_file <OWN_DEV_FILE> --test_file <OWN_TEST_FILE>
```
Note that this only work for baseline.py, enhanced_baseline.py, and LSTM.py, not for the Language Model.

