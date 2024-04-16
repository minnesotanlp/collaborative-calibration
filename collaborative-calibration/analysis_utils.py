from typing import List

import numpy as np
from sklearn.metrics import brier_score_loss, auc, roc_curve, RocCurveDisplay, roc_auc_score

import matplotlib.pyplot as plt


######### Metrics #########

def interval_ece(mask_all, conf_all, M=20):
    bin_boundaries = np.linspace(0, 1, M + 1)
    ece = 0
    prop_all_bins, accuracy_all_bins = [], []
    for m in range(M):
        in_bin, mask_in_bin, conf_in_bin = [], [], []
        bin_lower, bin_upper = bin_boundaries[m], bin_boundaries[m + 1]
        for i, conf in enumerate(conf_all):
            is_in = conf > bin_lower and conf <= bin_upper
            in_bin.append(float(is_in))
            if is_in:
                mask_in_bin.append(mask_all[i])
                conf_in_bin.append(conf)
        # |Bm|/n
        prop_in_bin = np.mean(in_bin)
        prop_all_bins.append(prop_in_bin)

        if prop_in_bin > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = np.mean(mask_in_bin)
            accuracy_all_bins.append(accuracy_in_bin)
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = np.mean(conf_in_bin)
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_all_bins.append(0)

    return ece, prop_all_bins, accuracy_all_bins, bin_boundaries


def brier_score(mask_all, conf_all):
    return brier_score_loss(mask_all, conf_all)


######### Visualization #########

def plot_reliability_diagram(
    bin_boundaries: List[float], confidence_all: List[float], accuracy_all: List[float], title: str
) -> None:
    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    facecolors = ["blue" for _ in accuracy_all]
    cmap = list(zip(facecolors, confidence_all / np.sum(confidence_all)))
    plt.bar(
        bin_boundaries[:-1],
        accuracy_all,
        width=1 / len(bin_boundaries),
        align="edge",
        color=cmap,
        edgecolor="blue",
    )
    plt.plot(bin_boundaries, bin_boundaries, color="r", linestyle="--")
    plt.xlabel("confidence Bin")
    plt.ylabel("Accuracy Bin")
    plt.title(title)
    plt.show()

def plot_majority_confidences(pre_conf_all, pre_majority_freq_all, post_conf_all, post_majority_freq_all):
    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle('Pre- and Post-deliberation confidence from the majority case')
    axes[0].scatter(pre_conf_all, pre_majority_freq_all, s=3)
    axes[0].plot([0, 1], [0, 1], '--', color='r', alpha=0.25, zorder=0)
    axes[0].set_xlabel('Verbal', labelpad = 5)
    axes[0].set_ylabel('Frequency', labelpad = 5)
    axes[1].scatter(post_conf_all, post_majority_freq_all, s=3)
    axes[1].plot([0, 1], [0, 1], '--', color='r', alpha=0.25, zorder=0)
    axes[1].set_xlabel('Verbal', labelpad = 5)
    axes[1].set_ylabel('Frequency', labelpad = 5)
    plt.show()

def plot_roc(y_true, y_score, estimator):
    plt.clf()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=estimator)
    display.plot()
    plt.title(f"auroc {roc_auc_score(y_true, y_score)}")
    plt.show()
