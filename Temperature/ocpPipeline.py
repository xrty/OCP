import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os

def fix_randomness(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def get_qhat_ordinal_aps(prediction_function, cal_scores, cal_labels, alpha, lamda, tol=1e-6):
    n = cal_scores.shape[0]
    left, right = 0.001, 1.999
    best_q = right
    target_coverage = np.ceil((n + 1) * (1 - alpha)) / n
    while right - left > tol:
        mid = (left + right) / 2
        coverage, *_ = evaluate_sets(prediction_function, np.copy(cal_scores), np.copy(cal_labels), mid, alpha, lamda)
        if coverage >= target_coverage:
            best_q = mid
            right = mid
        else:
            left = mid
    return best_q

def sliding_window_predict_set(val_scores, qhat, lamda):
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
                score = prob_sum - lamda * dist_penalty
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
        ###
        else:
            P[i, 0:K] = True
    return P

def evaluate_sets(prediction_function, val_scores, val_labels, qhat, alpha, lamda, print_bool=False):
    sets = prediction_function(val_scores, qhat, lamda)
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
    if print_bool:
        print(f"alpha: {alpha} | coverage: {coverage:.4f} | avg size: {sizes.mean():.4f} | qhat: {qhat:.4f}")
    return coverage, label_stratified_coverage, sizes_distribution, sizes.mean(), label_distribution

def run_pipeline(scores, labels, alpha=0.1, num_trials=10, lamda_list=None, save_dir='result'):
    os.makedirs(save_dir, exist_ok=True)
    if lamda_list is None:
        lamda_list = [0.001, 0.005, 0.01, 0.02, 0.05]

    for lamda in lamda_list:
        print(f"\n==== Running lambda = {lamda} ====")
        results = []
        #fix_randomness(seed=1000)
        prediction_function = sliding_window_predict_set

        for trial in tqdm(range(num_trials)):
            perm = np.load(f"files/perm_trial{trial}.npy")
            shuffled_scores = scores[perm]
            shuffled_labels = labels[perm]
            n = shuffled_scores.shape[0] // 2
            cal_scores, val_scores = shuffled_scores[:n], shuffled_scores[n:]
            cal_labels, val_labels = shuffled_labels[:n], shuffled_labels[n:]
            est_labels = np.argmax(shuffled_scores, axis=1)
            acc = (shuffled_labels == est_labels).mean()

            qhat = get_qhat_ordinal_aps(prediction_function, np.copy(cal_scores), np.copy(cal_labels), alpha, lamda)
            coverage, _, _, avg_size, _ = evaluate_sets(
                prediction_function, np.copy(val_scores), np.copy(val_labels),
                qhat, alpha, lamda, print_bool=True
            )

            results.append({
                'alpha': alpha,
                'coverage': coverage,
                'average_size': avg_size,
                'qhat': qhat
            })

        avg_avg_size = np.mean([r['average_size'] for r in results])
        results.append({'avg_avg_size': avg_avg_size})

        df = pd.DataFrame(results)
        csv_name = os.path.join(save_dir, f'result_lamda{lamda}.csv')
        df.to_csv(csv_name, index=False)
        print(f"Saved to {csv_name}")

if __name__ == "__main__":
    scores = np.load('files/score.npy') 
    labels = np.load('files/label.npy').astype(int)
    scores = scores.reshape(-1, scores.shape[-1])
    labels = labels.flatten()
    valid = (scores.sum(axis=1) > 0) & (labels >= 0)
    scores = scores[valid]
    labels = labels[valid]
    run_pipeline(scores, labels, alpha=0.1, num_trials=10,
                 lamda_list=[0.0011,0.0013,0.0015,0.0017,0.0019],
                 save_dir='resultPart3')
