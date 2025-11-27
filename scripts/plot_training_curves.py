import numpy as np
import matplotlib.pyplot as plt

# Skip header row, read columns by name
data = np.genfromtxt("../outputs/DM-CNN_training_metrics_20251101-07_20_PM_0.001_AG_GN_LM_TRAINING.csv", delimiter=",", names=True)

# Extract columns
step        = data['step']
train_loss  = data['train_loss']
test_loss   = data['test_loss']
train_accu  = data['train_accu']
test_accu   = data['test_accu']

# Plot Accuracy
plt.figure(figsize=(10, 4))
plt.plot(step, train_accu, label='Train Accuracy', color='blue')
plt.plot(step, test_accu, label='Test Accuracy', color='green')
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/accuracy_plot.png")
plt.show()

# Plot Loss
plt.figure(figsize=(10, 4))
plt.plot(step, train_loss, label='Train Loss', color='orange')
plt.plot(step, test_loss, label='Test Loss', color='red')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/loss_plot.png")
plt.show()

