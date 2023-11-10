import matplotlib.pyplot as plt
import seaborn as sns

# Data for each drug
data = {
    "Amphetamine": {
        "confusion_matrix": [[317, 59], [49, 47]],
        "classification_report": {"precision": [0.84, 0.49], "recall": [0.87, 0.44], "f1-score": [0.85, 0.47], "accuracy": 0.77}
    },
    "Benzodiazepines": {
        "confusion_matrix": [[288, 88], [42, 54]],
        "classification_report": {"precision": [0.82, 0.59], "recall": [0.73, 0.36], "f1-score": [0.77, 0.45], "accuracy": 0.73}
    },
    "Cannabis": {
        "confusion_matrix": [[158, 53], [47, 214]],
        "classification_report": {"precision": [0.76, 0.80], "recall": [0.77, 0.82], "f1-score": [0.76, 0.81], "accuracy": 0.79}
    },
    "Cocaine": {
        "confusion_matrix": [[357, 88], [9, 18]],
        "classification_report": {"precision": [0.80, 0.17], "recall": [0.98, 0.17], "f1-score": [0.88, 0.27], "accuracy": 0.79}
    },
    "Ecstasy (MDMA)": {
        "confusion_matrix": [[278, 61], [58, 75]],
        "classification_report": {"precision": [0.83, 0.55], "recall": [0.82, 0.56], "f1-score": [0.82, 0.56], "accuracy": 0.75}
    },
    "Ketamine": {
        "confusion_matrix": [[414, 53], [1, 4]],
        "classification_report": {"precision": [0.99, 0.07], "recall": [1.00, 0.07], "f1-score": [0.99, 0.07], "accuracy": 0.89}
    },
    "Legal High": {
        "confusion_matrix": [[274, 45], [68, 85]],
        "classification_report": {"precision": [0.86, 0.65], "recall": [0.86, 0.56], "f1-score": [0.86, 0.60], "accuracy": 0.76}
    },
    "Lysergic acid diethylamide (LSD)": {
        "confusion_matrix": [[311, 62], [61, 40]],
        "classification_report": {"precision": [0.83, 0.39], "recall": [0.84, 0.40], "f1-score": [0.84, 0.39], "accuracy": 0.78}
    },
    "Methamphetamine (Crystal Meth)": {
        "confusion_matrix": [[377, 7], [82, 6]],
        "classification_report": {"precision": [0.98, 0.46], "recall": [0.98, 0.07], "f1-score": [0.98, 0.12], "accuracy": 0.81}
    },
    "Mushrooms (Shrooms)": {
        "confusion_matrix": [[308, 50], [46, 68]],
        "classification_report": {"precision": [0.86, 0.58], "recall": [0.86, 0.60], "f1-score": [0.86, 0.59], "accuracy": 0.80}
    },
    "Nicotine": {
        "confusion_matrix": [[202, 73], [64, 133]],
        "classification_report": {"precision": [0.73, 0.68], "recall": [0.76, 0.65], "f1-score": [0.75, 0.66], "accuracy": 0.71}
    },
}


# Function to plot confusion matrix and classification report
def plot_results(name, cm, report):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix for {name}')
    ax[0].set_xlabel('Predicted labels')
    ax[0].set_ylabel('True labels')
    ax[0].xaxis.set_ticklabels(['Non-User', 'User'])
    ax[0].yaxis.set_ticklabels(['Non-User', 'User'])

    # Plotting classification report
    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    categories = ['Non-User', 'User']
    for metric in metrics:
        for i, category in enumerate(categories):
            ax[1].bar(metric, report[metric][i], label=f'{category} ({metric[0].upper()})', alpha=0.5)
    ax[1].set_title(f'Classification Report for {name}')
    ax[1].set_ylabel('Scores')
    ax[1].set_ylim([0, 1])
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# Plotting for each drug
for drug, values in data.items():
    plot_results(drug, values["confusion_matrix"], values["classification_report"])