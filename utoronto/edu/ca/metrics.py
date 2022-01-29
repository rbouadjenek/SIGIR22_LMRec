# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import warnings

from sklearn import metrics
import numpy as np
from sklearn.utils._encode import _unique, _encode

def mean_reciprocal_rank(y_true, y_score):
    one_hot = np.eye(len(y_score[0]))[y_true]
    temp = np.argsort(-1 * y_score, axis=1)
    ranks = temp.argsort() + 1
    scores = 1/np.sum(one_hot*ranks, axis=1)
    return np.mean(scores), (1.645 * np.std(scores)) / np.sqrt(len(scores))

def top_k_accuracy_score(y_true, y_score, k=2):
    """Top-k Accuracy classification score.
    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.
    Read more in the :ref:`User Guide <top_k_accuracy_score>`
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers). The binary case expects
        scores with shape (n_samples,) while the multiclass case expects scores
        with shape (n_samples, n_classes). In the multiclass case, the order of
        the class scores must correspond to the order of ``labels``, if
        provided, or else to the numerical or lexicographical order of the
        labels in ``y_true``.
    k : int, default=2
        Number of most likely outcomes considered to find the correct label.
    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.
    See also
    --------
    accuracy_score
    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
    ...                     [0.3, 0.4, 0.2],  # 1 is in top 2
    ...                     [0.2, 0.4, 0.3],  # 2 is in top 2
    ...                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
    >>> top_k_accuracy_score(y_true, y_score, k=2)
    0.75
    >>> # Not normalizing gives the number of "correctly" classified samples
    >>> top_k_accuracy_score(y_true, y_score, k=2, normalize=False)
    3
    """
    classes = [x for x in range(len(y_score[0]))]
    y_true_encoded = _encode(y_true, uniques=classes)
    sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
    hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)
    return np.mean(hits), (1.645 * np.std(hits)) / np.sqrt(len(hits))


def get_performance(y_true: list, y_pred: list):
    precision_score = metrics.precision_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
    recall_score = metrics.recall_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
    f1_score = metrics.f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
    hit_rates5 = top_k_accuracy_score(y_true, y_pred, k=5)
    hit_rates10 = top_k_accuracy_score(y_true, y_pred, k=10)
    hit_rates20 = top_k_accuracy_score(y_true, y_pred, k=20)
    accuracy = top_k_accuracy_score(y_true, y_pred, k=1)
    mrr = mean_reciprocal_rank(y_true, y_pred)
    out = {'Precision': precision_score, 'Recall': recall_score, 'F1-Score': f1_score, 'Accuracy': accuracy,
           'MRR': mrr, 'HR@5': hit_rates5, 'HR@10': hit_rates10, 'HR@20': hit_rates20}
    return out


if __name__ == '__main__':
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
                        [0.3, 0.4, 0.2],  # 1 is in top 2
                        [0.2, 0.4, 0.3],  # 2 is in top 2
                        [0.7, 0.2, 0.1]])  # 2 isn't in top 2
    print(get_performance(y_true, y_score))

