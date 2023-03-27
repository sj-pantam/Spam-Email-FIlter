import numpy as np
import os

# ===================== ABOUT THE DATA ========================
# Inside the 'data' folder, the emails are separated into 'train' 
# and 'test' data. Each of these folders has nested 'spam' and 'ham'
# folders, each of which has a collection of emails as txt files.
# You will only use the emails in the 'train' folder to train your 
# classifier, and will evaluate on the 'test' folder.
        
# The emails we are using are a subset of the Enron Corpus,
# which is a set of real emails from employees at an energy
# company. The emails have a subject line and a body, both of
# which are 'tokenized' so that each unique word or bit of
# punctuation is separated by a space or newline. The starter
# code provides a function that takes a filename and returns a
# set of all of the distinct tokens in the file.
# =============================================================

class NaiveBayes():
    """
    This is a Naive Bayes spam filter, that learns word spam probabilities 
    from our pre-labeled training data and then predicts the label (ham or spam) 
    of a set of emails that it hasn't seen before.
    """
    def __init__(self):
        """
        These variables are described in the 'fit' function below. 
        You will also need to access them in the 'predict' function.
        """
        self.num_train_hams = 0
        self.num_train_spams = 0
        self.word_counts_spam = {}
        self.word_counts_ham = {}
        self.HAM_LABEL = 'ham'
        self.SPAM_LABEL = 'spam'

    def load_data(self, path:str='data/'):
        """
        This function loads all the train and test data and returns
        the filenames as lists. You do not need to worry about how this
        function works unless you're curious.
        """
        assert set(os.listdir(path)) == set(['test', 'train'])
        assert set(os.listdir(os.path.join(path, 'test'))) == set(['ham', 'spam'])
        assert set(os.listdir(os.path.join(path, 'train'))) == set(['ham', 'spam'])

        train_hams, train_spams, test_hams, test_spams = [], [], [], []
        for filename in os.listdir(os.path.join(path, 'train', 'ham')):
            train_hams.append(os.path.join(path, 'train', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'train', 'spam')):
            train_spams.append(os.path.join(path, 'train', 'spam', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'ham')):
            test_hams.append(os.path.join(path, 'test', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'spam')):
            test_spams.append(os.path.join(path, 'test', 'spam', filename))

        return train_hams, train_spams, test_hams, test_spams

    def word_set(self, filename:list):
        """ 
        This function reads in a file and returns a set of all 
        the words. It ignores the subject line.
        """
        with open(filename, 'r') as f:
            text = f.read()[9:] # Ignoring 'Subject:'
            text = text.replace('\r', '')
            text = text.replace('\n', ' ')
            words = text.split(' ')
            return set(words)

    def fit(self, train_hams:list, train_spams:list):

        self.num_train_hams = len(train_hams)
        self.num_train_spams = len(train_spams)

        for filename in train_hams:
            set_words = self.word_set(filename)
            for word in set_words:
                if word in self.word_counts_ham:
                    self.word_counts_ham[word] += 1
                else:
                    self.word_counts_ham[word] = 1

        for filename in train_spams:
            set_words = self.word_set(filename)
            for word in set_words:
                if word in self.word_counts_spam:
                    self.word_counts_spam[word] += 1
                else:
                    self.word_counts_spam[word] = 1

    def predict(self, filename:str):

        ham_prob = 0
        spam_prob = 0
        set_words = self.word_set(filename)
        for word in set_words:
            word_count_ham = self.word_counts_ham.get(word, 0)
            word_count_spam = self.word_counts_spam.get(word, 0)
            ham_prob += np.log((word_count_ham + 1) / (self.num_train_hams + 2))
            spam_prob += np.log((word_count_spam + 1) / (self.num_train_spams + 2))

        if ham_prob >= spam_prob:
            return self.HAM_LABEL
        else:
            return self.SPAM_LABEL


    def accuracy(self, hams:list, spams:list):

        total_correct = 0
        total_datapoints = len(hams) + len(spams)
        for filename in hams:
            if self.predict(filename) == self.HAM_LABEL:
                total_correct += 1
        for filename in spams:
            if self.predict(filename) == self.SPAM_LABEL:
                total_correct += 1
        return total_correct / total_datapoints

if __name__ == '__main__':
    # Create a Naive Bayes classifier.
    nbc = NaiveBayes()

    # Load all the train/test ham/spam data.
    train_hams, train_spams, test_hams, test_spams = nbc.load_data()

    # Fit the model to the training data.
    nbc.fit(train_hams, train_spams)

    # Print out the accuracy on the train and test sets.
    print("Train Accuracy: {}".format(nbc.accuracy(train_hams, train_spams)))
    print("Test  Accuracy: {}".format(nbc.accuracy(test_hams, test_spams)))
