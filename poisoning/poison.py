# Matthew Jagielski

import math
import sys
import numpy as np
import numpy.linalg as la
from functools import partial

# import my modules
from my_args import setup_argparse 
from flip import inf_flip, alfa_tilt, adaptive, rand_flip, rand_flip_nobd, rmml
from gd_poisoners import *


def open_dataset(f,visualize):
  if visualize:
    rng = np.random.RandomState(1)
    random_state = 1
    x, y = make_regression(n_samples=300, n_features=1, random_state=random_state, noise=15.0, bias=1.5)
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())

    plt.plot(x, y, 'k.')
    
    global colmap
    colmap = {}
  else:
    x,y = read_dataset_file(f)

  return x, y


def read_dataset_file(f):
  with open(f) as dataset:
    x = []
    y = []
    cols = dataset.readline().split(',')
    print(cols)
    
    global colmap
    colmap = {}
    for i, col in enumerate(cols):
      if ':' in col:
        label = col.split(':')[0]
        if label in colmap:
          colmap[label].append(i-1) # why i-1?
        else:
          colmap[label] = [i-1]
    for line in dataset:
      line = [float(val) for val in line.split(',')]
      y.append(line[0])
      x.append(line[1:])

    return np.array(x), np.array(y)


def open_logging_files(logdir,modeltype,logind,args):
  myname = str(modeltype)+str(logind)
  logdir = logdir + os.path.sep + myname
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  with open(os.path.join(logdir,'cmd'),'w') as cmdfile:
    cmdfile.write(' '.join(['python3'] + sys.argv))
    cmdfile.write('\n')
    for arg in args.__dict__:
      cmdfile.write('{}: {}\n'.format(arg,args.__dict__[arg]))

  trainfile = open(logdir + os.path.sep + "train.txt",'w')
  testfile = open(logdir + os.path.sep + "test.txt",'w')
  validfile = open(logdir + os.path.sep + "valid.txt",'w')
  resfile = open(logdir + os.path.sep + "err.txt",'w')
  resfile.write('poisct,itnum,obj_diff,obj_val,val_mse,test_mse,time\n')
  return trainfile,testfile,validfile,resfile,logdir


def sample_dataset(x, y, trnct, poisct, tstct, vldct, seed):
    size = x.shape[0]
    print(size)
    print(trnct, tstct, vldct, poisct)
    
    np.random.seed(seed)
    fullperm = np.random.permutation(size)

    sampletrn = fullperm[:trnct]
    sampletst = fullperm[trnct:trnct + tstct]
    samplevld = fullperm[trnct + tstct:trnct + tstct + vldct]
    samplepois = np.random.choice(size, poisct, replace=True)

    trnx = x[sampletrn]
    trny = y[sampletrn]
    
    tstx = x[sampletst]
    tsty = y[sampletst]
    
    poisx = x[samplevld]
    poisy = y[samplevld]

    vldx = x[samplepois]
    vldy = y[samplepois]
    
    return trnx, trny, tstx, tsty, poisx, poisy, vldx, vldy


def roundpois(poisx,poisy):
  return np.around(poisx), np.where(poisy < 0.5, 0, 1)


# ------------------------------------------------------------------------------- 
# #datasets = ["icmldataset.txt",'contagio-preprocessed-missing.csv','pharm-preproc.csv','loan-processed.csv','house-processed.csv']
# ------------------------------------------------------------------------------- 
def main(args):
    trainfile, testfile, validfile, resfile, newlogdir =\
        open_logging_files(args.logdir, args.model, args.logind, args)
    x,y = open_dataset(args.dataset, args.visualize)
    trainx, trainy, testx, testy, poisx, poisy, validx, validy = \
        sample_dataset(x, y, args.trainct, args.poisct, args.testct, args.validct,\
                       args.seed)
    
    for x, y in zip(testx, testy, strict=True):
        testfile.write(str(y) + ',')
        testfile.write(','.join(map(str, x)) + '\n')
    testfile.close()
    
    for x, y in zip(validx, validy, strict=True):
        validfile.write(str(y) + ',')
        validfile.write(','.join(map(str, x)) + '\n')
    validfile.close()

    for x, y in zip(trainx, trainy, strict=True):
        trainfile.write(str(y) + ',')
        trainfile.write(','.join(map(str, x)) + '\n')

    print(la.matrix_rank(trainx))
    print(trainx.shape)

    totprop = args.poisct/(args.poisct + args.trainct) # p / (n + p)
    print(totprop)

    inits = {
        'alfatilt': alfa_tilt,
        'inflip': inf_flip,
        'adaptive': adaptive,
        'randflip': rand_flip,
        'randflipnobd': rand_flip_nobd,
        'rmml': partial(rmml, colmap=colmap),
    }

    types = {
        'linreg': LinRegGDPoisoner,
        'lasso': LassoGDPoisoner,
        'enet': ENetGDPoisoner,
        'ridge': RidgeGDPoisoner,
    }

    init = inits[args.initialization]

    genpoiser = types[args.model](trainx, trainy, testx, testy, validx, validy,
                                  args.eta, args.beta, args.sigma, args.epsilon,
                                  args.multiproc, trainfile, resfile,
                                  args.objective,args.optimizey, colmap)

    timestart,timeend = None, None
    bestpoisx, bestpoisy, besterr = None, None, -1
    
    for initit in range(args.numinit):
        numsamples = math.ceil(args.trainct * totprop / (1 - totprop))
        poisx, poisy = init(trainx, trainy, numsamples)
        clf, _ = genpoiser.learn_model(
          np.concatenate((trainx, poisx), axis=0),
          np.concatenate((trainy, poisy), axis=0),
          None
        )
        err = genpoiser.computeError(clf)[0]
        print("Validation Error:", err)
        if err > besterr:
            bestpoisx, bestpoisy, besterr = np.copy(poisx), np.copy(poisy), err
    poisx, poisy = bestpoisx, bestpoisy
    poiser = types[args.model](trainx, trainy, testx, testy, validx, validy,\
                               args.eta, args.beta, args.sigma, args.epsilon,\
                               args.multiproc, trainfile, resfile,\
                               args.objective, args.optimizey, colmap)

    
    for i in range(args.partct + 1):
        # curprop = k * totprop, k = 1/n, 2/n, ..., n/n
        curprop = (i + 1) * totprop / (args.partct + 1)
        # if prop is c, then c = p / (n + p) => p = nc / (1 - c)
        numsamples = math.ceil(args.trainct * curprop / (1 - curprop))
        curpoisx = poisx[:numsamples,:]
        curpoisy = poisy[:numsamples]
        trainfile.write("\n")

        timestart = datetime.datetime.now()
        poisres, poisresy = poiser.poison_data(curpoisx, curpoisy, timestart, args.visualize, newlogdir)
        poisedx = np.concatenate((trainx, poisres), axis=0)
        poisedy = np.concatenate((trainy, poisresy))

        clfp, _ = poiser.learn_model(poisedx,poisedy,None)
        clf = poiser.initclf
        if args.rounding:
            roundx,roundy = roundpois(poisres,poisresy)
            rpoisedx = np.concatenate((trainx, roundx), axis=0)
            rpoisedy = np.concatenate((trainy, roundy))
            clfr, _ = poiser.learn_model(rpoisedx,rpoisedy,None)
            rounderr = poiser.computeError(clfr)

        errgrd = poiser.computeError(clf)
        err = poiser.computeError(clfp)

        timeend = datetime.datetime.now()

        towrite = [numsamples,-1,None,None,err[0],err[1],(timeend-timestart).total_seconds()]
        resfile.write(','.join([str(val) for val in towrite])+"\n")
        trainfile.write("\n")
        for x, y in zip(poisres, poisresy, strict=True):
            trainfile.write(str(y) + '\n')
            trainfile.write(','.join(map(str, x)) + '\n')

        if args.rounding:
            towrite = [numsamples,'r',None,None,rounderr[0],rounderr[1],(timeend-timestart).total_seconds()]
            resfile.write(','.join([str(val) for val in towrite])+"\n")
            trainfile.write("\nround\n")
            for x, y in zip(roundx, roundy, strict=True):
                trainfile.write(str(y) + '\n')
                trainfile.write(','.join(map(str, x)) + '\n')

        resfile.flush()
        trainfile.flush()
        os.fsync(resfile.fileno())
        os.fsync(trainfile.fileno())
   
    trainfile.close()
    testfile.close()

    print()
    print("Unpoisoned")
    print("Validation MSE:",errgrd[0])
    print("Test MSE:",errgrd[1])
    print('Poisoned:')
    print("Validation MSE:",err[0])
    print("Test MSE:",err[1])
    if args.rounding:
        print("Rounded")
        print("Validation MSE",rounderr[0])
        print("Test MSE:", rounderr[1])



if __name__=='__main__':
    print("starting poison ...\n")
    parser = setup_argparse()
    args = parser.parse_args()

    print("-----------------------------------------------------------")
    print(args)
    print("-----------------------------------------------------------")
    main(args)
