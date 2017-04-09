import load_files as lf
import vectorizer as vec
import libsvm
import numpy as np
import xgboost as xgb

# Uncomment this if you're running it for the first time
# dim = 50
# traindata = lf.load_csv('train.csv')
# testdata = lf.load_csv('test.csv')
# glove = lf.load_glove('glove.6B.' + str(dim) + 'd.txt')
# train_vec = vec.vectorize(traindata, glove, dim)
# np.save("train_vec", train_vec)
# test_vec = vec.vectorize(testdata, glove, dim, is_train=False)
# np.save("test_vec", test_vec)

# train_vec = np.load("train_vec.npy")
# test_vec = np.load("test_vec.npy")
# libsvm.convert_to_libsvm(train_vec, True)
# libsvm.convert_to_libsvm(test_vec, False)
# libsvm.split_to_libsvm(train_vec)
#
# read in data
# dtrain = xgb.DMatrix('datalib.txt.train')
# dtest = xgb.DMatrix('datalib.txt.test')
# # specify parameters via map
# param = {'booster': 'dart',
#          'max_depth': 5, 'learning_rate': 0.05,
#          'objective': 'binary:logistic', 'silent': False,
#          'sample_type': 'weighted',
#          'normalize_type': 'forest',
#          'rate_drop': 0.05,
#          'skip_drop': 0.5}
# num_round = 3
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# # ntree_limit must not be 0
# preds = bst.predict(dtest, ntree_limit=num_round)
# for pred in preds:
#     print(pred)

lf.conv_to_csv("pred.txt")