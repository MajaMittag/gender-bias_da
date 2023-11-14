######################################################
# Utility functions                                  #
######################################################

# Imports 
from danlp.datasets import DKHate
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
import string
from typing import List
from wordcloud import WordCloud 

# Set seed
SEED = 42

######################################################

def load_dkhate(test_size:float) -> pd.Series:
    """Load DKHate and split into train and test set based on provided test size.

    Args:
        test_size (float): proportion of data used for test split 

    Returns:
        4 x pd.Series: returns X_train, X_test, y_train, y_test
    """
    
    # Load train and test split
    test_hate, train_hate = DKHate().load_with_pandas()

    # Concatenate train and test and split back into train and test to get desired split
    all_hate = pd.concat([train_hate, test_hate])
    all_hate.rename(columns={'subtask_a': 'label'}, inplace=True) # rename last column (subtask_a --> label)
    all_hate.replace({"NOT":0, "OFF":1}, inplace=True) # make labels numeric
    X_train, X_test, y_train, y_test = train_test_split(all_hate['tweet'], all_hate['label'], test_size=test_size, random_state=SEED)
    
    return X_train, X_test, y_train, y_test

######################################################

def oversample_data(X_train:pd.Series, y_train:pd.Series, strategy:float, smote:bool) -> pd.Series:
    """Oversamples the minority class using the provided sampling strategy.

    Args:
        X_train (pd.Series): original training data
        y_train (pd.Series): original training labels
        strategy (float): sampling strategy. E.g. if it's 0.5, then you'll end up with a 1:2 distribution, whereas 1.0 results in a 1:1 distribution and so on.
        smote (bool): whether to use SMOTE or random sampling.

    Returns:
        2 x pd.Series: returns oversampled versions of X_train, X_test, y_train, y_test
    """
    
    if smote:
        oversample = SMOTE(sampling_strategy=strategy)
    else:
        oversample = RandomOverSampler(sampling_strategy=strategy)
    
    X_train_oversampled, y_train_oversampled = oversample.fit_resample(X_train, y_train)
    
    return X_train_oversampled, y_train_oversampled

######################################################

def create_wordcloud(X_train:pd.Series, y_train:pd.Series, mask:int):
    """Generates a word cloud using the texts where the label == mask.

    Args:
        X_train (pd.Series): training data
        y_train (pd.Series): training labels
        mask (int): label mask, e.g. 1

    Returns:
        WordCloud: the word cloud object
    """
    
    # get one long string of all the text
    words = ' '.join([x for x in X_train[y_train == mask]])
    
    # generate cloud
    cloud = WordCloud(width=800, height=400, background_color='white', random_state=SEED).generate(words)

    return cloud

######################################################

def plot_wordcloud(word_cloud, title:str) -> None:
    """Plots a given word cloud.

    Args:
        word_cloud (_type_): the word cloud object
        title (str): desired plot title
    """
    
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize = 15)
    plt.show()

######################################################

def preprocess(text:str, stopwords:list, to_string:bool=True):
    """Preprocesses data by lowercasing, removing punctuation and removing stop words. Can be returned as string (to_string=True) or list of tokens.

    Args:
        text (str): the text to be preprocessed
        stopwords (list): list of stop words
        to_string (bool, optional): whether to return preprocessed text as string. Defaults to True.

    Returns:
        str or list: string or list of preprocessed text
    """
    # lowercase text
    lowercase_text = text.lower()
    
    # remove punctuation
    re_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
    wo_punctuation = re_punctuation.sub('', lowercase_text)
    
    # remove digits
    clean_text = re.sub(r"[\d]", "", wo_punctuation)
    
    # # sub multiple occurrences of "user" in a row for only the first occurence
    # clean_text = re.sub(r"\buser\b(?:\s*user\s*){1,}", "user ", clean_text)
    # (performs the same with this enabled)
    
    # # split concatenated words that contain url by url
    # clean_text = ' '.join(re.split(r"(url)", clean_text))
    # (performs slightly worse with this enabled)
    
    # tokenize and remove stop words
    tokens = [token for token in clean_text.split() if token not in stopwords]
    
    if to_string:
        tokens = ' '.join(tokens)
    
    return tokens

######################################################

def get_vocab(column:pd.Series, are_tokens:bool=True) -> set:
    """Returns set of vocabulary in a pd.Series object.

    Args:
        column (pd.Series): the data
        are_tokens (bool, optional): whether the data is tokenized. Defaults to True.

    Returns:
        set: the unique vocabulary items
    """
    vocab = []
    for text in column:
        if are_tokens:
            for word in text:
                vocab.append(word)
        else:
            for word in text.split():
                vocab.append(word)
    return set(vocab)

######################################################

def plot_heatmap(confusion_matrix, title:str) -> None:
    """Plots heatmap of confusion matrix with the provided title.

    Args:
        confusion_matrix (_type_): the confusion matrix to plot
        title (str): the desired title
    """
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0.5,1.5], ['Non-toxic', 'Toxic'])
    plt.yticks([0.5,1.5], ['Non-toxic', 'Toxic'])
    plt.show()