## email-spam-detection

# about

This project uses K-Nearest Neighbors and Naive Bayes implemented from scratch to detect spam as described in the promp below.

# methods

1.  Use K-Fold Cross Validation to test K-Nearest Neighbors and Naive Bayes for given classification problem.


# usage

Extract `ham.zip` and `spam.zip` into two distinct folders labeled ham and spam.

```
python detect-spam.py <full path of ham directory> <full path of spam directory>

ex: 

python detect-spam.py c:\\enron1\ham\* c:\\enron1\spam\*"
```

Be sure to include the *

After executing the above once pickle files will be generated therefore subsequent runs can be done via:

```
python detect-spam.py
```

# prompt
![Prompt 1](https://github.com/tmatrixhy/ml-email-spam-detection-knn-naive-bayes/blob/master/prompt1.jpg)
![Prompt 2](https://github.com/tmatrixhy/ml-email-spam-detection-knn-naive-bayes/blob/master/prompt2.jpg)
![Prompt 3](https://github.com/tmatrixhy/ml-email-spam-detection-knn-naive-bayes/blob/master/prompt3.jpg)

# results

![Results](https://github.com/tmatrixhy/ml-email-spam-detection-knn-naive-bayes/blob/master/results.png)