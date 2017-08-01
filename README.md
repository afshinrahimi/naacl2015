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
@InProceedings{rahimi2017a,
  author    = {Rahimi, Afshin  and  Cohn, Trevor  and  Baldwin, Timothy},
  title     = {A Neural Model for User Geolocation and Lexical Dialectology},
  booktitle = {Proceedings of ACL-2017 (short papers) preprint},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics}
}
```

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>

