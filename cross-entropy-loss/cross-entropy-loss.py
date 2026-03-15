import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    # Select probabilities of the correct classes
    p_correct = y_pred[np.arange(len(y_true)), y_true]

    # Compute average cross entropy loss
    loss = -np.mean(np.log(p_correct))

    return loss