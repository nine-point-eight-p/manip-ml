import numpy as np
import bisect

"""
Code that seems never executed or not compatible to current version of the project.
"""


# TRIM algorithm
def robustopt(x, y, count, lam, poiser):
    length = x.shape[0]
    width = x.shape[1]
    y = np.array(y)
    tau = sorted(np.random.permutation(length))[:count]
    inittau = tau[:]
    clf = None

    newtau = []
    it = 0
    toterr = 10000
    lasterr = 20000

    clf, _ = poiser.learn_model(x, y, None)

    while sorted(tau) != sorted(newtau) and it < 400 and lasterr - toterr > 1e-5:
        newtau = tau[:]
        lasterr = toterr
        subx = x[tau]
        suby = y[tau]
        clf.fit(subx, suby)
        w, b = clf.coef_, clf.intercept_

        residvec = [(w * np.transpose(x[i]) + b - y[i]) ** 2 for i in range(length)]

        residtopns = sorted([(residvec[i], i) for i in range(length)])[:count]
        resid = [val[1] for val in residtopns]
        topnresid = [val[0] for val in residtopns]

        # set tau to indices of n largest values in error
        tau = sorted(resid)  # [1 if i in resid else 0 for i in range(length)]
        # recompute error
        toterr = sum(topnresid)
        it += 1
    return clf, w, b, lam, tau


def infflip(x, y, count, poiser):
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)
    allgains = []
    for i in range(x.shape[0]):
        posgain = (np.sum(hmat[i]) * (1 - y[i]), 1)
        neggain = (np.sum(hmat[i]) * y[i], 0)
        allgains.append(max(posgain, neggain))

    totalprob = sum([a[0] for a in allgains])
    allprobs = [0]
    for i in range(len(allgains)):
        allprobs.append(allprobs[-1] + allgains[i][0])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))
    gainsy = [allgains[ind][1] for ind in poisinds]

    # sortedgains = sorted(enumerate(allgains),key = lambda tup: tup[1])[:count]
    # poisinds = [a[0] for a in sortedgains]
    # bestgains = [a[1][1] for a in sortedgains]

    return x[poisinds], gainsy


def levflip(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)

    alllevs = [hmat[i, i] for i in range(x.shape[0])]
    totalprob = sum(alllevs)
    allprobs = [0]
    for i in range(len(alllevs)):
        allprobs.append(allprobs[-1] + alllevs[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if clf.predict(x[curind].reshape(1, -1)) < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


def cookflip(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    preds = [clf.predict(x[i].reshape(1, -1)) for i in range(x.shape[0])]
    errs = [(y[i] - preds[i]) ** 2 for i in range(x.shape[0])]
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)

    allcooks = [hmat[i, i] * errs[i] / (1 - hmat[i, i]) ** 2 for i in range(x.shape[0])]

    totalprob = sum(allcooks)

    allprobs = [0]
    for i in range(len(allcooks)):
        allprobs.append(allprobs[-1] + allcooks[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if clf.predict(x[curind].reshape(1, -1)) < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


def farthestfirst(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    preds = [clf.predict(x[i].reshape(1, -1)) for i in range(x.shape[0])]
    errs = [(y[i] - preds[i]) ** 2 for i in range(x.shape[0])]
    totalprob = sum(errs)
    allprobs = [0]
    for i in range(len(errs)):
        allprobs.append(allprobs[-1] + errs[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if preds[curind] < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


def alfatilt(x, y, count, poiser):
    trueclf, _ = poiser.learn_model(x, y, None)
    truepreds = trueclf.predict(x)

    goalmodel = np.random.uniform(
        low=-1 / sqrt(x.shape[1]), high=1 / sqrt(x.shape[1]), shape=(x.shape[1] + 1)
    )
    goalpreds = np.dot(x, goalmodel[:-1]) + goalmodel[-1].item()

    svals = np.square(trueclf.predict(x) - y)  # squared error
    svals = svals / svals.max()
    qvals = np.square(goalpreds - y)
    qvals = qvals / qvals.max()

    flipscores = (svals + qvals).tolist()

    totalprob = sum(flipscores)
    allprobs = [0]
    allpoisy = []
    for i in range(len(flipscores)):
        allprobs.append(allprobs[-1] + flipscores[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))
        if truepreds[curind] < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy
