import matplotlib.pyplot as plt
import seaborn as sns

# Data for each drug
data = {
    "Amphetamine": {
        "confusion_matrix": [[273, 22], [54, 28]],
        "classification_report": {"precision": [0.83, 0.56], "recall": [0.93, 0.34], "f1-score": [0.88, 0.42], "accuracy": 0.80}
    },
    "Benzodiazepines": {
        "confusion_matrix": [[234, 28], [74, 41]],
        "classification_report": {"precision": [0.76, 0.59], "recall": [0.89, 0.36], "f1-score": [0.82, 0.45], "accuracy": 0.73}
    },
    "Cannabis": {
        "confusion_matrix": [[141, 28], [46, 162]],
        "classification_report": {"precision": [0.75, 0.85], "recall": [0.83, 0.78], "f1-score": [0.79, 0.81], "accuracy": 0.80}
    },
    "Cocaine": {
        "confusion_matrix": [[283, 15], [61, 18]],
        "classification_report": {"precision": [0.82, 0.55], "recall": [0.95, 0.23], "f1-score": [0.88, 0.32], "accuracy": 0.80}
    },
    "Ecstasy (MDMA)": {
        "confusion_matrix": [[233, 40], [57, 47]],
        "classification_report": {"precision": [0.80, 0.54], "recall": [0.85, 0.45], "f1-score": [0.83, 0.49], "accuracy": 0.74}
    },
    "Ketamine": {
        "confusion_matrix": [[333, 1], [41, 2]],
        "classification_report": {"precision": [0.89, 0.67], "recall": [1.00, 0.05], "f1-score": [0.94, 0.09], "accuracy": 0.89}
    },
    "Legal High": {
        "confusion_matrix": [[229, 32], [52, 64]],
        "classification_report": {"precision": [0.81, 0.67], "recall": [0.88, 0.55], "f1-score": [0.85, 0.60], "accuracy": 0.78}
    },
    "Lysergic acid diethylamide (LSD)": {
        "confusion_matrix": [[271, 27], [43, 36]],
        "classification_report": {"precision": [0.86, 0.57], "recall": [0.91, 0.46], "f1-score": [0.89, 0.51], "accuracy": 0.81}
    },
    "Methamphetamine (Crystal Meth)": {
        "confusion_matrix": [[294, 17], [50, 16]],
        "classification_report": {"precision": [0.85, 0.48], "recall": [0.95, 0.24], "f1-score": [0.90, 0.32], "accuracy": 0.82}
    },
    "Mushrooms (Shrooms)": {
        "confusion_matrix": [[260, 24], [49, 44]],
        "classification_report": {"precision": [0.84, 0.65], "recall": [0.92, 0.47], "f1-score": [0.88, 0.55], "accuracy": 0.81}
    },
    "Nicotine": {
        "confusion_matrix": [[106, 63], [46, 162]],
        "classification_report": {"precision": [0.70, 0.72], "recall": [0.63, 0.78], "f1-score": [0.66, 0.75], "accuracy": 0.71}
    },
}

# Function to plot all results in one figure
def plot_all_confusion_matrices(data):
    num_drugs = len(data)
    num_rows = -(-num_drugs // 3)  # Ceiling division to get the number of rows
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (drug, values) in enumerate(data.items()):
        sns.heatmap(values["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(drug)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        axes[i].xaxis.set_ticklabels(['Non-User', 'User'])
        axes[i].yaxis.set_ticklabels(['Non-User', 'User'])

    # Hide any unused subplots
    for j in range(i + 1, num_rows * 3):
        axes[j].axis('off')

    plt.show()

# Function to plot all classification reports in a 1 by 3 format
def plot_all_classification_reports(data):
    num_drugs = len(data)
    num_rows = -(-num_drugs // 3)  # Ceiling division to get the number of rows
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (drug, values) in enumerate(data.items()):
        if i < num_rows * 3 - 1:  # Ensure we don't plot on the last subplot
            report = values["classification_report"]
            metrics = ['precision', 'recall', 'f1-score']
            for metric in metrics:
                axes[i].bar(metric, report[metric][0], alpha=0.8)
                axes[i].bar(metric, report[metric][1], alpha=0.8)
            axes[i].bar('accuracy', report['accuracy'], color='green')
            axes[i].set_title(drug)
            axes[i].set_ylim([0, 1])
            axes[i].legend(['Non-User', 'User', 'Accuracy'], loc='lower right')
        else:
            break 

    # Hide any unused subplots before the last one
    for j in range(i, num_rows * 3 - 1):
        axes[j].axis('off')

    plt.show()

# Plotting all confusion matrices and classification reports
plot_all_confusion_matrices(data)
plot_all_classification_reports(data)