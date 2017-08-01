'''
Created on 27 Jul 2017

@author: af
'''
import numpy as np
import argparse
import sys
import logging
import random
from haversine import haversine
from data import DataLoader
import mlp

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.random.seed(77)
random.seed(77)

def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    '''
    given gold labels and predicted labels find the distance between
    the gold location of user and the median training point in the 
    predicted region.
    '''
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    gold_locations = []
    pred_locations = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        gold_locations.append((lat, lon))
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        pred_locations.append((lat_pred, lon_pred))  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    return np.mean(distances), np.median(distances), acc_at_161

def load_data(args):
    #remove @-mentions
    #token_pattern = r'(?u)(?<![@])\b\w+\b'
    #keep both @-mentions and hashtags with their signs
    token_pattern = r'(?u)@?#?\b\w\w+\b'
    dtype = 'float32'
    dl = DataLoader(data_home=args.dir, bucket_size=args.bucket, encoding=args.enc, 
                    celebrity_threshold=args.celebrity, one_hot_labels=False, 
                    mindf=args.mindf, maxdf=0.1, norm='l2', idf=True, btf=True, tokenizer=None, subtf=True, stops='english', token_pattern=token_pattern)
    dl.load_data()
    #uncomment if you want to draw the training points on a map using basemap
    #dataset_name = args.dir.split('/')[-3]
    #dl.draw_training_points(filename='points_{}.png'.format(dataset_name), world=True if 'world' in dataset_name else False, figsize=(4,3))
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    X_train = dl.X_train.astype(dtype)
    X_dev = dl.X_dev.astype(dtype)
    X_test = dl.X_test.astype(dtype)
    dl.assignClasses()
    Y_test = dl.test_classes.astype('int32')
    Y_train = dl.train_classes.astype('int32')
    Y_dev = dl.dev_classes.astype('int32')
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}
    
    #for regression
    '''
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype=dtype)
    loc_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype=dtype)
    loc_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype=dtype)
    Y_train = loc_train
    Y_dev = loc_dev
    Y_test = loc_test
    '''
    
    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    return data

def train(data, args):

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    classifier = mlp.MLP(n_epochs=args.epochs, batch_size=args.batch, complete_prob=False, add_hidden=True, 
                         regul_coef=args.reg, save_results=False, hidden_layer_size=args.hid, 
                         drop_out_coef=args.drop, early_stopping_max_down=5)
    classifier.fit(X_train, Y_train, X_dev, Y_dev)
    logging.info('dev results')
    mean, median, acc_at_161 = geo_eval(Y_dev, classifier.f_predict(X_dev), U_dev, classLatMedian, classLonMedian, userLocation)
    logging.info('test results')
    mean, median, acc_at_161 = geo_eval(Y_test, classifier.f_predict(X_test), U_test, classLatMedian, classLonMedian, userLocation)
    

def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-batch', metavar='int', help='SGD batch size', type=int, default=1000)
    parser.add_argument('-celebrity', metavar='int', help='minimum threshold for celebrities', type=int, default=5)
    parser.add_argument('-epochs', metavar='int', help='max epochs for training', type=int, default=100)
    parser.add_argument('-hid', metavar='int', help='Hidden layer size', type=int, default=500)
    parser.add_argument('-mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument('-dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument('-enc', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument('-reg', metavar='float', help='regularization coefficient)', type=float, default=1e-6)
    parser.add_argument('-drop', metavar='float', help='dropout coef default 0.5', type=float, default=0.5)
    parser.add_argument('-tune', action='store_true', help='if exists tune hyperparameters')
    parser.add_argument('-m', '--message', type=str) 
    
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    #THEANO_FLAGS='device=cuda0' nice -n 10 python textclassification.py -dir ~/datasets/cmu/processed_data/ -enc latin1 -reg 1e-5 -drop 0.5 -mindf 10 -hid 800 -batch 500
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    data = load_data(args)
    train(data, args)