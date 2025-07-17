import csv
import pdb
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    random.seed(seed)
    

def get_qhat_ordinal_aps(prediction_function, cal_scores, cal_labels, alpha, tol=1e-6):
    n = cal_scores.shape[0]
    left, right = 0.001, 0.999
    best_q = right
    target_coverage = np.ceil((n + 1) * (1 - alpha)) / n
    while right - left > tol:
        mid = (left + right) / 2
        coverage, _, _, _, _ = evaluate_sets(prediction_function, np.copy(cal_scores), np.copy(cal_labels), mid, alpha)
        if coverage >= target_coverage:
            best_q = mid
            right = mid
        else:
            left = mid
    return best_q


def ordinal_aps_prediction(val_scores, qhat):
    # if qhat > 1:  # bug somewhere?
        # return np.ones_like(val_scores).astype(bool)
    P = val_scores == val_scores.max(axis=1)[:, None]
    idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat # Places where top-1 isn't correct
    while idx_construction_incomplete.sum() > 0:
        P_inc = P[idx_construction_incomplete]
        scores_inc = val_scores[idx_construction_incomplete]
        set_cumsum = P_inc.cumsum(axis=1)
        lower_edge_idx = (P_inc > 0).argmax(axis=1)
        upper_edge_idx = set_cumsum.argmax(axis=1)
        
        # Where the lower edge is both valid and also has a higher softmax score than the upper edge
        lower_edge_wins = (
            ((lower_edge_idx - 1) >= 0) &
            (
                (upper_edge_idx + 1 > scores_inc.shape[1] - 1) |
                (
                    scores_inc[np.arange(scores_inc.shape[0]), np.maximum(lower_edge_idx - 1, 0)] > 
                    scores_inc[np.arange(scores_inc.shape[0]), np.minimum(upper_edge_idx + 1, scores_inc.shape[1] - 1)]
                )
            )
        )
        P_inc[lower_edge_wins, lower_edge_idx[lower_edge_wins] - 1] = True
        P_inc[~lower_edge_wins, upper_edge_idx[~lower_edge_wins] + 1] = True  # IndexError here when alpha is too small
        P[idx_construction_incomplete] = P_inc
        idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat
    return P

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
                score = prob_sum
                #score = prob_sum
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
        else:
            P[i, 0:K] = True

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

                if score >= qhat:
                    if u - l < best_len:
                        best_len = u - l
                        best_l, best_u = l, u

        if best_l != -1 and best_u != -1:
            P[i, best_l:best_u + 1] = True

    return P


def evaluate_sets(prediction_function, val_scores, val_labels, qhat, alpha, print_bool=False):
    sets = prediction_function(val_scores, qhat)
    sizes = sets.sum(axis=1)
    sizes_distribution = np.array([(sizes == i).mean() for i in range(5)])
    covered = sets[np.arange(val_labels.shape[0]), val_labels]
    coverage = covered.mean()
    label_stratified_coverage = [
        covered[val_labels == j].mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    label_distribution = [
        (val_labels == j).mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    if print_bool == True:
        print(f"alpha: {alpha} | coverage: {coverage:.4f} | average size: {sizes.mean():.4f} | qhat: {qhat:.4f}")
    return coverage, label_stratified_coverage, sizes_distribution, sizes.mean(), label_distribution

if __name__ == "__main__":
    results = []
    fix_randomness(seed=1000)
    # Experimental parameters
    num_trials = 10
    # Define miscoverage rate
    alpha = 0.1
    # Scores is n_patients X num_locations X num_severity_classes
    scores = np.load('files/score.npy')
    # Labels is n_patients X num_locations 
    labels = np.load('files/label.npy').astype(int)
    scores = scores.reshape(-1,scores.shape[-1])
    labels = labels.flatten()
    print(scores.shape)
    # Check validity
    valid = (scores.sum(axis=1) > 0) & (labels >= 0)
    scores = scores[valid]
    labels = labels[valid]

    ###
    # Version of conformal
    #prediction_function = ordinal_aps_prediction
    prediction_function = sliding_window_predict_set
    #prediction_function = brute_force_predict_set

    coverages = []
    for trial in range(num_trials):
    # Permute
        perm = np.random.permutation(scores.shape[0])
        scores = scores[perm]
        labels = labels[perm]
    # Split
        n = scores.shape[0] // 2  # 50/50 split
        cal_scores, val_scores = (scores[:n], scores[n:])
        cal_labels, val_labels = (labels[:n], labels[n:])
    # Calculate accuracy
        est_labels = np.argmax(scores, axis=1)
        acc = (labels == est_labels).mean()
        # print(f"Model accuracy: {acc}")

    # Calculate quantile
        qhat = get_qhat_ordinal_aps(prediction_function, np.copy(cal_scores), np.copy(cal_labels), alpha)
        #cal_scores_values = score_function(np.copy(cal_scores), np.copy(cal_labels))
        #qhat = np.quantile(cal_scores_values, 1 - alpha, method='higher')
    
    # Calculate sets
        (coverage, label_stratified_coverage, sizes_distribution, avg_size, label_distribution) = evaluate_sets(
            prediction_function, np.copy(val_scores), np.copy(val_labels), qhat, alpha, print_bool=True
        )
        
    # 存入字典
        results.append({
            'alpha': alpha,
            'coverage': coverage,
            'average_size': avg_size,
            'qhat': qhat
        })
    avg_avg_size = np.mean([r['average_size'] for r in results])
    print(f"\nFinal average of average size (avg_avg_size): {avg_avg_size:.4f}")
    results.append({'avg_avg_size': avg_avg_size})

# 保存到CSV
    df = pd.DataFrame(results)
    ###
    df.to_csv('result/baselines.csv', index=False)
    print(np.histogram(coverages))


