"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# mypy: ignore-errors
# pylint: disable=W0223


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch import Tensor
from torchvision.transforms import Compose, Normalize, ToTensor

import torch.optim as optim
from flwr_datasets import FederatedDataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import time
import timeit

# pylint: disable=unsubscriptable-object

class Net(nn.Module):
    def __init__(self, input_dim: int = 56):
        super(Net, self).__init__()
        #self.layer1 = nn.Linear(input_dim, 128)
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


    
def load_data(partition_id: int):


    # print('partition_id --> ')
    # print(partition_id)
    
    #print('num_partitions --> ')
    #print(num_partitions)
    
    # Load dataset locally
    dataset_path = "data/Obfuscated-MalMem2022.csv"
    dataset = pd.read_csv(dataset_path)

   # # Split into partitions
    partition_size = len(dataset) 
    #//  num_partitions  # num_clients defined elsewhere
    
    # print('partition_size --> ')
    # print(partition_size)
    
    start_idx = partition_id * partition_size
    
    # print('start_idx --> ')
   #  print(start_idx)
    
    end_idx = start_idx + partition_size
    
    # print('end_idx --> ')
    # print(end_idx)
    
    client_partition = dataset.iloc[start_idx:end_idx] 

    # print('client_partition --> ')
    # print(client_partition)
    #dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    # print('categorical_cols --> ')
    # print(categorical_cols)

    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )
   
    #print("X_train")
    #print(X_train)
       
    #print("X_test")
    #print(X_test)
    
    #print("y_train")
    #print(y_train)
    
    #print("y_test")
    #print(y_test)
    
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )


    X_train = preprocessor.fit_transform(X_train)
    
    
    X_test = preprocessor.transform(X_test)

    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    #print('<-- Fim Class load_data  --> ')
    return train_loader, test_loader

def rocy(model, train_loader, test_loader, num_epochs=1):
    #=================================================================
    # print("<<<<< Training loop >>>>")
    
    import torch
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    
    # print("<<<<< Create a PyTorch DataLoader for training and testing >>>>")
    # Move the model to a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("<<<<<<<<<<< Initialize lists to store metrics >>>>>>>>>>>>>")

    # Initialize lists to store metrics
    train_epochs = []
    train_losses = []
    train_accuracies = []
    train_auc_scores = []
    train_precisions = []
    train_recalls = []
    test_losses = []
    test_accuracies = []
    test_auc_scores = []
    test_precisions = []
    test_recalls = []

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = num_epochs

    print("<<<<<<<<<<< Initialize lists to store metrics epochs >>>>>>>>>>>>>")

    for epoch in range(num_epochs):
        model.train()

        #========================================================
        train_loss_sum = 0.0
        num_correct_train = 0
        num_samples_train = 0
        y_true_train = []
        y_pred_train = []
        #========================================================

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()

            num_correct_train += (predicted == y_batch).sum().item()
            num_samples_train += y_batch.size(0)
            y_true_train.extend(y_batch.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            train_loss_sum += loss.item()

        # Calculate training metrics
        train_accuracy = num_correct_train / num_samples_train
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        train_precision = precision_score(y_true_train, y_pred_train, zero_division=0)
        train_recall = recall_score(y_true_train, y_pred_train, zero_division=0)

        # Calculate average training loss
        average_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        test_loss_sum = 0.0
        num_correct_test = 0
        num_samples_test = 0
        y_true_test = []
        y_pred_test = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                batch_loss = criterion(outputs, y_batch)

                predicted = (outputs > 0.5).float()

                num_correct_test += (predicted == y_batch).sum().item()
                num_samples_test += y_batch.size(0)
                y_true_test.extend(y_batch.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

                test_loss_sum += batch_loss.item()

        # Calculate testing metrics
        test_accuracy = num_correct_test / num_samples_test
        test_auc = roc_auc_score(y_true_test, y_pred_test)
        test_precision = precision_score(y_true_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_test, zero_division=0)

        # Calculate average testing loss
        average_test_loss = test_loss_sum / len(test_loader)
        
        # Gerando e exibindo a Matriz de Confusão
        conf_matrix = confusion_matrix(y_true_test, y_pred_test)

        # Print epoch-wise metrics
        print(f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {average_train_loss:.4f} - "
            f"Train Acc: {train_accuracy:.4f} - "
            f"Train AUC: {train_auc:.4f} - "
            f"Train Prec: {train_precision:.4f} - "
            f"Train Recall: {train_recall:.4f} - "
            f"Test Loss: {average_test_loss:.4f} - "
            f"Test Acc: {test_accuracy:.4f} - "
            f"Test AUC: {test_auc:.4f} - "
            f"Test Prec: {test_precision:.4f} - "
            f"Test Recall: {test_recall:.4f} - " 
            f"Confusion matrix: {conf_matrix}"
            )

        # Store metrics for later analysis or plotting
        train_epochs.append(epoch)

        train_losses.append(f"{average_train_loss:.4f}")
        test_losses.append(f"{average_test_loss:.4f}")

        train_accuracies.append(f"{train_accuracy:.4f}")
        test_accuracies.append(f"{test_accuracy:.4f}")

        train_auc_scores.append(f"{train_auc:.4f}")
        test_auc_scores.append(f"{test_auc:.4f}")

        train_precisions.append(f"{train_precision:.4f}")
        test_precisions.append(f"{test_precision:.4f}")

        train_recalls.append(f"{train_recall:.4f}")
        test_recalls.append(f"{test_recall:.4f}")
       
        
    import matplotlib.pyplot as plt

    
    # Gráfico de Loss
    #plt.figure(figsize=(18, 6))

    # Subplot para perda (Loss)
    plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(train_epochs, test_losses, label='Test Loss', color='red', marker='o')
    plt.title('Train Loss - Flower - CIC-MalMem-2022')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    
    # Ajustar layout e mostrar os gráficos
    plt.tight_layout()
    plt.show()
    
   
    # Subplot para acurácia (Accuracy)
    plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(train_epochs, test_accuracies, label='Test Accuracy', color='red', marker='o')
    plt.title('Train Accuracy  - Flower - CIC-MalMem-2022')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    # Ajustar layout e mostrar os gráficos
    plt.tight_layout()
    plt.show()
    
   
    # Subplot para AUC
    plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_auc_scores, label='Train AUC', color='blue', marker='o')
    plt.plot(train_epochs, test_auc_scores, label='Test AUC', color='red', marker='o')
    plt.title('Train AUC - Flower - CIC-MalMem-2022')
    plt.xlabel('Época')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # Ajustar layout e mostrar os gráficos
    plt.tight_layout()
    plt.show()
    
    
    # Subplot para Precisions
    plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_precisions, label='Train Precision', color='blue', marker='o')
    plt.plot(train_epochs, test_precisions, label='Test Precision', color='red', marker='o')
    plt.title('Train Precision - Flower - CIC-MalMem-2022')
    plt.xlabel('Época')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Ajustar layout e mostrar os gráficos
    plt.tight_layout()
    plt.show()
    
    plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_recalls, label='Train Recall', color='blue', marker='o')
    plt.plot(train_epochs, test_recalls, label='Test Recall', color='red', marker='o')
    plt.title('Train Recall - Flower - CIC-MalMem-2022')
    plt.xlabel('Época')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # Ajustar layout e mostrar os gráficos
    plt.tight_layout()
    plt.show()




def main():
    
    inicio = timeit.default_timer()
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("Centralized PyTorch training")
    
    print(" >>>>>>>>>> Submission of articles - IEEE Latin America Transactions <<<<<<<<<< ")
    #print("Load data")
    
    trainloader, testloader = load_data(0)
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    #train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    #train(net, trainloader, 10)
    #print("Evaluate model")
    #loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    #loss, accuracy = evaluate(net, testloader)
    
    rocy(net, trainloader, testloader, num_epochs=10)
    
    #print("Loss: ", loss)
    #print("Accuracy: ", accuracy)

    fim = timeit.default_timer()
    print ('duracao: %f' % (fim - inicio))

if __name__ == "__main__":
    main()


























