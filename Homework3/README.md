# Homework 3

The goal of this assignment is to translate a given source sentence (e.g., German) to a target sentence (e.g., English). You need to implement a Neural Machine Translation (NMT) system using a sequence-to-sequence (Seq2Seq) model with global attention mechansim (i.e., Luong attention'15). Please complete the blank code correctly. 
Specifically, I plan to list all students' BLEU scores and rank them in the highest order.

When grading your assignments, we will be doing it through your model's BLEU score on the test set. Specifically, we plan to enumerate all students' BLEU scores from their model and rank them in the highest order. So, if you want to get the high score, you have to improve your NMT system by applying other techniques (e.g., teacher forcing, stack more layers, other attention mechansim). If you have applied other methods to improve performance, please write a report with the reason. 

# Submission due

Due date: 11/15 (Sun), 23.59 p.m.
Please submit your code and report (if it is necessary) to KLMS.

# Usage
- Downloading the dataset
After completing all code, you can train your model using IWSLT 2014 De-En dataset as training data. The source and the target languages are German and English, respectively. Before training your model, you must download the IWSLT 2014 De-En dataset by excuting the following command:

wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
unzip iwslt2014_ende.zip

- Building the vocabulary
After downloading the dataset, please execute the following command to generate the vocabulary file for both source and target languages:

bash run.sh vocab

Note that you can change parameters or arguments to improve the quality of vocabulary.

- Training your model
After building the vocabulary file, please excute the following command to train your model:

bash run.sh train

Also, you can change parameters to obtain better performance on dev set.

- Testing your model
We will score your assignment through the BLEU score of your model on the test set. 

# Contact

If you have some questions, please contact to me via an email.
passing2961@gmail.com



