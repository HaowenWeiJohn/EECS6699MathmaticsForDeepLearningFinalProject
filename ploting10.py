import pickle

import matplotlib.pyplot as plt


file_name1 = "experiment_cifar10/Resnet2/history.pickle"
file_name2 = "experiment_cifar10/Resnet4/history.pickle"
file_name3 = "experiment_cifar10/Resnet6/history.pickle"
file_name4 = "experiment_cifar10/Resnet8/history.pickle"
file_name5 = "experiment_cifar10/Resnet10/history.pickle"
file_name6 = "experiment_cifar10/Resnet12/history.pickle"
file_name7 = "experiment_cifar10/Resnet14/history.pickle"
file_name8 = "experiment_cifar10/Resnet16/history.pickle"
file_name9 = "experiment_cifar10/Resnet18Original/history.pickle"




def read_result(file_name):
    with open(file_name, "rb") as f:
        history = pickle.load(f)

    val_accuracy = history["val_accuracy_history"]
    val_loss = ["val_loss_history"]
    return history


Resnet2 = read_result(file_name1)
Resnet4 = read_result(file_name2)
Resnet6 = read_result(file_name3)
Resnet8 = read_result(file_name4)
Resnet10 = read_result(file_name5)
Resnet12 = read_result(file_name6)
Resnet14 = read_result(file_name7)
Resnet16 = read_result(file_name8)
Resnet18Original = read_result(file_name9)

print(max(Resnet2['val_accuracy_history']))
print(max(Resnet4['val_accuracy_history']))
print(max(Resnet6['val_accuracy_history']))
print(max(Resnet8['val_accuracy_history']))
print(max(Resnet10['val_accuracy_history']))
print(max(Resnet12['val_accuracy_history']))
print(max(Resnet14['val_accuracy_history']))
print(max(Resnet16['val_accuracy_history']))
print(max(Resnet18Original['val_accuracy_history']))

label1 = 'ResNet-2'
label2 = 'ResNet-4'
label3 = 'ResNet-6'
label4 = 'ResNet-8'
label5 = 'ResNet-10'
label6 = 'ResNet-12'
label7 = 'ResNet-14'
label8 = 'ResNet-16'
label9 = 'ResNet-18'

plt.plot(Resnet2['val_accuracy_history'], label=label1)
plt.plot(Resnet4['val_accuracy_history'], label=label2)
plt.plot(Resnet6['val_accuracy_history'], label=label3)
plt.plot(Resnet8['val_accuracy_history'], label=label4)
plt.plot(Resnet10['val_accuracy_history'], label=label5)
plt.plot(Resnet12['val_accuracy_history'], label=label6)
plt.plot(Resnet14['val_accuracy_history'], label=label7)
plt.plot(Resnet16['val_accuracy_history'], label=label8)
plt.plot(Resnet18Original['val_accuracy_history'], label=label9)
plt.ylim(0, 100)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(Resnet2['val_loss_history'], label=label1)
plt.plot(Resnet4['val_loss_history'], label=label2)
plt.plot(Resnet6['val_loss_history'], label=label3)
plt.plot(Resnet8['val_loss_history'], label=label4)
plt.plot(Resnet10['val_loss_history'], label=label5)
plt.plot(Resnet12['val_loss_history'], label=label6)
plt.plot(Resnet14['val_loss_history'], label=label7)
plt.plot(Resnet16['val_loss_history'], label=label8)
plt.plot(Resnet18Original['val_loss_history'], label=label9)
plt.ylim(0, 0.03)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss")
plt.show()

