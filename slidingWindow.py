import numpy as np
import pandas as pd

def fix_randomness(seed=0):
    np.random.seed(seed)
    import random
    random.seed(seed) 

def new_score_function_prefix_sum(cal_scores, cal_labels):
    N, K = cal_scores.shape
    scores = np.zeros(N)
    yhat = np.argmax(cal_scores, axis=1)
    for i in range(N):
        y_pred = yhat[i]
        y_true = cal_labels[i]
        left = min(y_pred, y_true)
        right = max(y_pred, y_true)
        interval_sum = cal_scores[i, left:right+1].sum()
        distance = (abs(y_pred - y_true)) + 1
        ###
        #scores[i] = interval_sum + distance
        scores[i] = interval_sum
    return scores

def get_qhat_from_scores(cal_scores, cal_labels, alpha):
    prefix_scores = new_score_function_prefix_sum(cal_scores, cal_labels)
    n = len(prefix_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha))) + 1
    #rank = n - rank
    sorted_score = np.sort(prefix_scores)[::-1]
    qhat = sorted_score[rank]
    return qhat

def sliding_window_predict_set(val_scores, qhat):
    N, K = val_scores.shape
    P = np.zeros((N, K), dtype=bool)

    for i in range(N):
        f = val_scores[i]
        y_star = np.argmax(f)
        prefix = np.zeros(K + 1)
        for j in range(K):
            prefix[j + 1] = prefix[j] + f[j]
        best_len = float('inf')
        best_l = best_u = -1
        l = 0
        for u in range(y_star, K):
            while l <= y_star:
                prob_sum = prefix[u + 1] - prefix[l]
                dist_penalty = abs(y_star - l) + abs(y_star - u)
                #score = prob_sum + dist_penalty
                score = prob_sum

                if y_star >= l and y_star <= u and score >= qhat:
                    if u - l < best_len:
                        best_len = u - l
                        best_l, best_u = l, u

                if score >= qhat:
                    l += 1
                else:
                    break

        if best_l != -1 and best_u != -1:
            P[i, best_l:best_u + 1] = True

    return P

def brute_force_predict_set(val_scores, qhat):
    N, K = val_scores.shape
    P = np.zeros((N, K), dtype=bool)

    for i in range(N):
        f = val_scores[i]
        y_star = np.argmax(f)

        best_len = float('inf')
        best_l = best_u = -1

        for l in range(0, y_star + 1):
            for u in range(y_star, K):
                prob_sum = np.sum(f[l:u + 1])
                score = prob_sum
                #score += 0.15*(u-l)
                if score >= qhat:
                    if u - l < best_len:
                        best_len = u - l
                        best_l, best_u = l, u

        if best_l != -1 and best_u != -1:
            P[i, best_l:best_u + 1] = True

    return P

if __name__ == "__main__":
    fix_randomness(seed=1000)
    alpha = 0.1
    num_trials = 50
    results = []

    scores = np.load('../files/scores.npy')
    labels = np.load('../files/labels.npy').astype(int)
    scores = scores.reshape(-1, scores.shape[-1])
    labels = labels.flatten()

    valid = (scores.sum(axis=1) > 0) & (labels >= 0)
    scores = scores[valid]
    labels = labels[valid]

    raw_scores = scores.copy()
    raw_labels = labels.copy()

    for trial in range(num_trials):
        perm = np.random.permutation(raw_scores.shape[0])
        scores = raw_scores[perm]
        labels = raw_labels[perm]

        n = scores.shape[0] // 2
        cal_scores, val_scores = scores[:n], scores[n:]
        cal_labels, val_labels = labels[:n], labels[n:]
        qhat = get_qhat_from_scores(np.copy(cal_scores), np.copy(cal_labels), alpha)
        ###
        prediction_sets = sliding_window_predict_set(np.copy(val_scores), qhat)
        #prediction_sets = brute_force_predict_set(np.copy(val_scores), qhat)
        correct = prediction_sets[np.arange(len(val_labels)), val_labels]
        coverage = correct.mean()
        avg_size = prediction_sets.sum(axis=1).mean()
        results.append({'trial': trial, 'qhat': qhat, 'coverage': coverage, 'avg_size': avg_size})
    df = pd.DataFrame(results)
    print(f"Average coverage over {num_trials} trials: {df['coverage'].mean():.4f}")
    print(f"Average prediction set size over {num_trials} trials: {df['avg_size'].mean():.4f}")

    