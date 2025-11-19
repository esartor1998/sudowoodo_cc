import torch
import numpy as np
import sklearn.metrics as metrics
import mlflow
import pickle

from tqdm import tqdm

import torch

def blocked_matmul(mata, matb, threshold=None, k=None, batch_size=512):
    """
    MUCH faster GPU version of blocked top-k or threshold similarity search.
    """

    # Ensure torch tensors on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mata = torch.as_tensor(np.array(mata), device=device)
    matb = torch.as_tensor(np.array(matb), device=device)

    results = []

    for start in range(0, len(matb), batch_size):
        block = matb[start:start+batch_size]  # (B, D)

        # compute similarity (N1, B)
        sim_mat = mata @ block.T   # GPU matmul

        # --- TOP-K MODE -------------------------------------------------
        if k is not None:
            # sim_mat: (N1, B)
            # topk over dim=0 → top k rows for each column = top k matches for each element in block
            vals, idxs = torch.topk(sim_mat, k=k, dim=0)    # both shapes (k, B)

            # Build result list quickly
            for j in range(idxs.shape[1]):  # over block items
                b_index = start + j
                for i in range(k):
                    a_index = idxs[i, j].item()
                    score = vals[i, j].item()
                    results.append((a_index, b_index, score))

        # --- THRESHOLD MODE --------------------------------------------
        elif threshold is not None:
            mask = sim_mat >= threshold
            a_idx, b_rel = mask.nonzero(as_tuple=True)
            for a, b in zip(a_idx.tolist(), b_rel.tolist()):
                results.append((a, start + b, sim_mat[a, b].item()))

    return results

def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset, faster."""

    all_probs = []
    all_y = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            if model.task_type == 'em':
                x1, x2, x12, y = batch
                logits = model(x1, x2, x12)
            else:
                x, y = batch
                logits = model(x)

            probs = logits.softmax(dim=1)[:, 1]

            # accumulate as torch tensors (faster)
            all_probs.append(probs.cpu())
            all_y.append(y.cpu())

    # concatenate once (much faster)
    all_probs = torch.cat(all_probs).numpy()
    all_y = torch.cat(all_y).numpy()

    # ---------- FIXED THRESHOLD MODE ----------
    if threshold is not None:
        pred = (all_probs > threshold).astype(int)

        # dump results (same as before)
        pickle.dump(pred, open('test_results.pkl', 'wb'))
        mlflow.log_artifact('test_results.pkl')

        f1 = metrics.f1_score(all_y, pred)
        p = metrics.precision_score(all_y, pred)
        r = metrics.recall_score(all_y, pred)
        return f1, p, r

    # ---------- SEARCH BEST THRESHOLD ----------
    best_f1 = -1
    best_th = 0.5
    best_p = 0.0
    best_r = 0.0

    # vectorized threshold loop (fast)
    for th in np.arange(0.0, 1.0, 0.05):
        pred = (all_probs > th).astype(int)

        f1 = metrics.f1_score(all_y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_p = metrics.precision_score(all_y, pred)
            best_r = metrics.recall_score(all_y, pred)
            best_th = th

    return best_f1, best_p, best_r, best_th
