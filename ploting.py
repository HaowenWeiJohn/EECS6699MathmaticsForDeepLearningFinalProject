import pickle

import matplotlib.pyplot as plt

file_name1 = "experiment_cifar10/Resnet18ACT_BN_ADD/history.pickle"
file_name2 = "experiment_cifar10/Resnet18BN_ACT_ADD/history.pickle"
file_name3 = "experiment_cifar10/Resnet18Expand/history.pickle"
file_name4 = "experiment_cifar10/Resnet18ExpandConcat/history.pickle"
file_name5 = "experiment_cifar10/Resnet18Original/history.pickle"
file_name6 = "experiment_cifar10/Resnet18OriginalDropOut/history.pickle"
file_name7 = "experiment_cifar10/Resnet18OriginalPlane/history.pickle"
file_name8 = "experiment_cifar10/Resnet18OriginalSwapBNACT/history.pickle"


def read_result(file_name):
    with open(file_name, "rb") as f:
        history = pickle.load(f)

    val_accuracy = history["val_accuracy_history"]
    val_loss = ["val_loss_history"]
    return history


Resnet18ACT_BN_ADD = read_result(file_name1)
Resnet18BN_ACT_ADD = read_result(file_name2)
Resnet18Expand = read_result(file_name3)
Resnet18ExpandConcat = read_result(file_name4)
Resnet18Original = read_result(file_name5)
Resnet18OriginalDropOut = read_result(file_name6)
Resnet18OriginalPlain = read_result(file_name7)
Resnet18OriginalSwapBNACT = read_result(file_name8)

print(max(Resnet18ACT_BN_ADD['val_accuracy_history']))
print(max(Resnet18BN_ACT_ADD['val_accuracy_history']))
print(max(Resnet18Expand['val_accuracy_history']))
print(max(Resnet18ExpandConcat['val_accuracy_history']))
print(max(Resnet18Original['val_accuracy_history']))
print(max(Resnet18OriginalDropOut['val_accuracy_history']))
print(max(Resnet18OriginalPlain['val_accuracy_history']))
print(max(Resnet18OriginalSwapBNACT['val_accuracy_history']))

label1 = 'ResNet-Dropout-18'
label2 = 'ResNet-18'

plt.plot(Resnet18OriginalDropOut['val_accuracy_history'], label=label1)
plt.plot(Resnet18Original['val_accuracy_history'], label=label2)
plt.ylim(0, 100)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(Resnet18OriginalDropOut['val_loss_history'], label=label1)
plt.plot(Resnet18Original['val_loss_history'], label=label2)
plt.ylim(0, 0.03)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss")
plt.show()
