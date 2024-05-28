import numpy as np
import sklearn.linear_model as lm


# Huber regression
def huber(x, y, epss):
    scores = []
    bestclf, besteps, bestscore = None, None, 0

    for eps in epss:
        clf = lm.HuberRegressor(epsilon=eps, max_iter=10000, alpha=1e-5)
        clf.fit(x, y)
        score = clf.score(x[~clf.outliers_], y[~clf.outliers_])
        scores.append(score)
        if score > bestscore:
            bestclf, besteps, bestscore = clf, eps, score

    return bestclf, besteps, scores


# Random Sample Consensus (RANSAC)
def ransac(x, y, model, lam, counts):
    allmodels = {
        "lasso": lm.Lasso(alpha=lam, max_iter=10000),
        "ridge": lm.Ridge(alpha=lam),
        "enet": lm.ElasticNet(alpha=lam, max_iter=10000),
        "linreg": lm.Ridge(alpha=0.00001),
    }

    scores = []
    bestclf, bestcount, bestscore = None, None, 0

    for count in counts:
        clfransac = lm.RANSACRegressor(allmodels[model], min_samples=count)
        clfransac.fit(x, y)
        score = clfransac.score(x[clfransac.inlier_mask_], y[clfransac.inlier_mask_])
        scores.append(score)
        if score > bestscore:
            bestclf, bestcount, bestscore = clfransac, count, score

    return bestclf, bestcount, scores


# TRIM
def trim(x, y, count, model, lam, n_iter=400):
    clf, _ = model.learn_model(x, y, None, lam)
    indices = list(range(x.shape[0]))
    last_err = float("inf")

    for _ in range(n_iter):
        subx = x[indices]
        suby = y[indices]
        clf.fit(subx, suby)
        residual = (clf.predict(x) - y) ** 2

        top_residual_indices = np.argpartition(residual, -count)[-count:]
        top_residual = residual[top_residual_indices]

        # set inds to indices of n largest values in error
        new_indices = sorted(top_residual_indices)
        # recompute error
        cur_err = np.sum(top_residual)

        if new_indices == indices or last_err - cur_err < 1e-5:
            break

        indices = new_indices
        last_err = cur_err

    return clf, lam, indices


# a: [F], b: [..., F]
def trimmed_inner_product(a, b, n):
    prods = a * b
    indices = np.argpartition(prods, n, axis=-1)[..., :n]
    corr = np.sum(np.take_along_axis(prods, indices, axis=-1) ** 2)
    selfcorr = np.sum(np.take_along_axis(b, indices, axis=-1) ** 2)
    return corr / selfcorr


def chen_pred(chen_model, x):
    return np.dot(x, chen_model)


# Robust Thresholding Regression (RoTR)
def chen_rotr(x, y, count, ks):
    print(x.shape, y.shape, count, ks)

    # x: [B, F], y: [F]
    corrs = trimmed_inner_product(y, x, count)
    top_corr_indices = np.argsort(np.abs(corrs))[::-1]
    bestmodel, besterrs, bestk = None, float("inf"), 0
    allerrs = []

    for k in ks:
        indices = top_corr_indices[:k]
        model = corrs[indices]

        preds = chen_pred(model, x)
        curerr = np.mean((preds - y) ** 2)
        allerrs.append(curerr)

        if curerr < besterrs:
            bestmodel = model
            besterrs = curerr
            bestk = k

    return np.array(bestmodel), bestk, allerrs


# Reject On Negative Impact (RONI)
def roni(x, y, count, model, lam, trainsizes):
    RONI_TRIALS = 5
    RONI_VALID_SIZE = 50

    allerrs = []
    bestclf, bestsize, bestcleaninds, bestscore = None, None, [], 0

    for trainsize in trainsizes:
        print(trainsize)
        all_increases = []

        for i in range(x.shape[0]):
            curx, cury = x[i], y[i]
            cur_increase = 0

            for j in range(RONI_TRIALS):
                # sample train, valid set
                traininds = np.random.choice(x.shape[0], size=trainsize)
                validinds = np.random.choice(x.shape[0], size=RONI_VALID_SIZE)

                trainx, trainy = x[traininds], y[traininds]
                validx, validy = x[validinds], y[validinds]

                # train on both train and train plus point
                exclude, _ = model.learn_model(trainx, trainy, None, lam)
                include, _ = model.learn_model(
                    np.append(trainx, curx.reshape((1, -1)), axis=0),
                    np.append(trainy, cury.reshape((1,)), axis=0),
                    None,
                    lam,
                )

                exclude_valid = exclude.predict(validx)
                exclude_mse = np.mean((exclude_valid - validy) ** 2)

                include_valid = include.predict(validx)
                include_mse = np.mean((include_valid - validy) ** 2)

                cur_increase += include_mse - exclude_mse

            all_increases.append(cur_increase)

        # get smallest <count> increases and retrain
        clean_indices = np.argpartition(all_increases, count)[:count]
        clean_x = x[clean_indices]
        clean_y = y[clean_indices]
        clean_model, clean_lam = model.learn_model(clean_x, clean_y, None, lam)
        score = clean_model.score(clean_x, clean_y)
        allerrs.append(score)

        if score > bestscore:
            bestclf = clean_model
            bestsize = trainsize
            bestcleaninds = clean_indices
            bestscore = score

    return bestclf, bestsize, bestcleaninds, bestscore, allerrs
