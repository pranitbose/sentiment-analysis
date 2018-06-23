# Sentiment Analysis
Sentiment Analysis of product based reviews using Machine Learning Approaches.

## Description
This project aims to perform sentiment classification of online product reviews using various Machine Learning classifiers. This project analyzes sentiment on dataset from document level (review level). Data used in this project are online product reviews collected from _amazon.com_. The Amazon reviews dataset used in this project consists of reviews from _amazon_. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review. For more information, please refer to the following paper: _J. McAuley_ and _J. Leskovec_. The final dataset is constructed by randomly taking 200,000 samples for each review score from 1 to 5. In total there are 1,000,000 samples. This project involves comparative study of the performance of 4 Machine Learning classifier models - Multinomial Naïve Bayes, Logistic Regression, Linear SVC and Random Forest. The best classifier was chosen to standardize the model to classify any product reviews in the future with promising outcomes. The user review taken as input is classified using the chosen model with respect to sentiment classes/categories - Postive and Negative, based on the Sentimental Orientation of the opinions it contains.

## Prerequisites
Make sure you have the following list of dependencies for this project installed and setup on your system first:

- Unix/Linux Operating System (Recommended but not necessarry)
- Python 3.6+
- Anaconda Distribution 5.2+
- NLTK Toolkit 3.3+

Some hardware requirements should also be fulfilled to run this project smoothly:

- At least 8GB RAM
- At least 50GB of usable Hard Disk space

## Usage
First download the project as zip archive and extract it to your desired location or just clone the repository using,

```
$ git clone https://github.com/pranitbose/sentiment-analysis.git
```

Donwload the dataset using the link provided in the _dataset_link.txt_ within the _datasets_ directory. Move the the downloaded dataset or whichever dataset you want to use into the _datasets_ directory. In case you are using your own dataset, you have to modify the filenames in the source code of _main.py_ to the one you'll be using. Many lines of the source code are commented on purpose and the state of the project is pickled wherever necessary to save computing resource and speed up the execution process eliminating repeatition of same steps more than once. A boolean variable named _do\_pickle_ is provided in _main.py_ to switch pickling on/off in the entire file by changing it's value in only one place in \__main__.

You only need to execute _main.py_ in your terminal to run this project. For the first run, you should uncomment the lines of the source code in _main.py_ and _sentiment\_analyzer.py_ which are already commented. Read the source code carefully before you do so. You should also enable pickling. All these will generate bunch of pickled files and various graph plots as a result of first execution. From following execution you should comment or uncomment the source code as per your requirements.

## License
This project is licensed under the terms of the MIT license.