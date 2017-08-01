Geolocation of Social Media Users Utilising both Text and Network Information 
---------------------------------------------------------------------------------

Text-based Model
----------------
first download the 3 datasets and put them in separate directories (e.g. cmu, na, world).
Note that you should have three gzipped files in each dataset (user_info.train.gz, user_info.dev.gz, user_info.test.gz)


To run the LR model use:

```
nice -n 10 python textclassification.py -dir ~/datasets/cmu/ -enc latin1 -reg 5e-5  -mindf 10  -model lr

```

This will use scikit-learn's SGDClassifier to classify each user in a location. You can control the regularisation
coefficient and minimum term document frequency.


Note that cmu dataset's encoding is 'latin1' while the other two datasets are 'utf-8'. 

parameters
----------
The two main parameters are regularization coefficient and encoding which should be set for each dataset as:

1. cmu: ``-reg 5e-5 -enc latin1``

2. na aka TwitterUS, WORLD aka TwitterWorld: ``-reg 1e-6 -enc utf-8``


Multilayer Perceptron Model
---------------------------
In this implementation (although not included in the paper), it is possible to use
a multilayer perceptron as classifier by using ``-model mlp`` in the parameters instead
of ``-model lr`` for logistic regression but then extra parameters such as batch size, dropout rate,
hidden layer size and regularization coefficients should be tuned.

I'd set the dropout to 0.5 and hidden size to 1000 for cmu, 2000 for na and 3000 for world
and then tune the regularization coefficient.





Geolocation datasets
--------------------
We experiment with three Twitter geolocation datasets
available at https://github.com/utcompling/textgrounder.
If you have any problems regarding the datasets contact me.


Usage
-----



```
usage: geodare.py [-h] [-dataset str] [-model str] [-datadir DATADIR] [-tune]

optional arguments:
  -h, --help            show this help message and exit
  -dataset str, --dataset str
                        dataset name (cmu, na, world)
  -model str, --model str
                        dialectology model (mlp, lr, word2vec)
  -datadir DATADIR      directory where input datasets (cmu.pkl, na.pkl,
                        world.pkl) are located.
  -tune                 if true, tune the hyper-parameters.
```

The preprocessed pickle files (e.g. na.pkl) are the vectorized version of
the geolocation datasets and are available upon request.

Citation
--------
```
@InProceedings{rahimi2015exploiting,
author="Rahimi, Afshin and Vu, Duy and Cohn, Trevor and Baldwin, Timothy",
title="Exploiting Text and Network Context for Geolocation of Social Media Users",
booktitle="Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
year="2015",
publisher="Association for Computational Linguistics",
pages="1362--1367",
location="Denver, Colorado",
url="http://aclweb.org/anthology/N15-1153"
}
```

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>

