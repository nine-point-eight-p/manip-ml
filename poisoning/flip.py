import numpy as np
from sklearn import linear_model


def inf_flip(X_tr, Y_tr, count):
    perturb = 1e-8 * np.identity(X_tr.shape[1])
    inv_cov = np.linalg.inv(X_tr.T @ X_tr + perturb)
    H = X_tr @ inv_cov @ X_tr.T # Projection matrix

    bests = np.sum(H, axis=1)
    room = 0.5 + np.abs(Y_tr - 0.5)
    stat = bests * room
    yvals = 1 - np.floor(0.5 + Y_tr)

    totalprob = np.sum(stat)
    allprobs = np.cumsum(stat)
    poisvals = np.random.uniform(0, totalprob, count)
    poisinds = np.searchsorted(allprobs, poisvals, side="left")

    return X_tr[poisinds], yvals[poisinds]


def alfa_tilt(X_tr, Y_tr, count):
    perturb = 1e-8 * np.identity(X_tr.shape[1])
    inv_cov = np.linalg.inv(X_tr.T @ X_tr + perturb)
    H = X_tr @ inv_cov @ X_tr.T

    randplane = np.random.standard_normal(size=X_tr.shape[1] + 1)
    w, b = randplane[:-1], randplane[-1]
    preds = np.dot(X_tr, w) + b
    yvals = preds.clip(0, 1)
    yvals = 1 - np.floor(0.5 + yvals)
    diff = yvals - Y_tr
    changes = np.dot(diff, H)
    changes = np.maximum(changes, 0)

    totalprob = np.sum(changes)
    allprobs = np.cumsum(totalprob)
    poisvals = np.random.uniform(0, totalprob, count)
    poisinds = np.searchsorted(allprobs, poisvals, side="left")

    return X_tr[poisinds], yvals[poisinds]


def adaptive(X_tr, Y_tr, count):
    X_tr_copy = np.copy(X_tr)
    Y_tr_copy = np.copy(Y_tr)

    room = 0.5 + np.abs(Y_tr_copy)
    yvals = 1 - np.floor(0.5 + Y_tr_copy)
    diff = (yvals - Y_tr_copy).ravel()

    X_pois = np.empty((count, X_tr.shape[1]))
    Y_pois = []
    for i in range(count):
        perturb = np.full((X_tr.shape[1], X_tr.shape[1]), 1e-8)
        inv_cov = np.linalg.inv(perturb + np.dot(X_tr_copy.T, X_tr_copy))
        H = np.dot(np.dot(X_tr_copy, inv_cov), X_tr_copy.T)

        bests = np.sum(H, axis=1)
        stat = np.multiply(bests.ravel(), diff)
        indtoadd = np.random.choice(
            stat.shape[0], p=np.abs(stat) / np.sum(np.abs(stat))
        )

        X_pois[i] = X_tr_copy[indtoadd, :]
        X_tr_copy = np.delete(X_tr_copy, indtoadd, axis=0)
        diff = np.delete(diff, indtoadd, axis=0)
        Y_pois.append(yvals[indtoadd])
        yvals = np.delete(yvals, indtoadd, axis=0)

    return X_pois, np.array(Y_pois)


def rand_flip(X_tr, Y_tr, count):
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    print("Points selected: ", poisinds)
    X_pois = X_tr[poisinds]
    # Y_pois = [1-Y_tr[i] for i in poisinds]  # this is for validating yopt, not for initialization
    Y_pois = (1 - Y_tr[poisinds]) > 0.5  # this is the flip all the way implementation
    return X_pois, Y_pois.astype(int)


def rand_flip_nobd(X_tr, Y_tr, count):
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    print("Points selected: ", poisinds)
    X_pois = X_tr[poisinds]
    Y_pois = 1 - Y_tr[poisinds]  # this is for validating yopt, not for initialization
    # Y_pois = [1 if 1-Y_tr[i]>0.5 else 0 for i in poisinds]  # this is the flip all the way implementation
    return X_pois, Y_pois


def rmml(X_tr, Y_tr, count, colmap):
    mean = np.mean(X_tr, axis=0)
    covar = np.dot((X_tr - mean).T, (X_tr - mean)) / X_tr.shape[0] + 0.01 * np.eye(
        X_tr.shape[1]
    )

    model = linear_model.Ridge(alpha=0.01)
    model.fit(X_tr, Y_tr)

    allpoisx = np.random.multivariate_normal(mean, covar, size=count)
    allpoisx = allpoisx >= 0.5
    poisy = model.predict(allpoisx)
    poisy = 1 - poisy
    poisy = poisy >= 0.5

    # Process group attributes
    for col in colmap.values():
        # Get columns with the highest value (top column) for each poisx elem
        indices = np.argmax(allpoisx[:, col], axis=1)
        topcols = np.asarray(col)[indices]
        topvals = np.take_along_axis(allpoisx, topcols[:, None], axis=1)

        # Set all columns of current group to 0
        allpoisx[:, col] = False
        # Set the top column to 1 if reaching a threshold
        newvals = topvals > 1 / (1 + len(col))
        np.put_along_axis(allpoisx, topcols[:, None], newvals, axis=1)

    return allpoisx.astype(int), poisy.astype(int) # Convert bool to int
