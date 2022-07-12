import os
import gc

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import numpy as np
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm
import TRNmodule


# My libraries
from Models.AttentionBaselineClassifier import AttentionBaseline
from Models.AvgClassifier import NeuralNetAvgPooling
from Models.MultiHeadAttention import MultiHeadAttentionModule, Encoder, MultiHeadAttentionClassifier, \
    MultiHeadAttentionClassifierPyTorch
from utils.feature_loaders import FeaturesDataset
from utils.utils import multiclassAccuracy, getFileName, saveMetrics, getBasePath, getFolderName
from utils.args import parser


def main():
    # Parse arguments
    args = parser.parse_args()
    args = vars(args)
    args["results_location"] = "RESULTS_AGGREGATION"
    args["base_path"] = getBasePath(__file__)
    # Get the name of the file and folder for saving results
    file_name: str = getFileName(args)
    results_path: str = getFolderName(args)

    # Define parameters
    model_class, parameters, external_parameters = instantiateModels(args)
    # Do grid search for finding best parameters

    total_models: int = len(ParameterGrid(external_parameters)) * len(ParameterGrid(parameters))
    i: int = 0
    for external_config in ParameterGrid(external_parameters):

        args["early"] = external_config["early"]
        args["weight_decay"] = external_config["weight_decay"]
        for config in ParameterGrid(parameters):
            if args["verbose"]:
                print(f"[GENERAL]Starting model {i + 1}/{total_models}")
                print("[GENERAL] Testing", config)
                memory_used = 0 if not torch.cuda.is_available() else torch.cuda.memory_allocated()
                print(f"[MEMORY] The memory alloocated in GPU is {memory_used}")
            args["config"] = (config, external_config)
            # Instantiate model with the configuration specified
            model_instance = model_class(**config)

            accuracy_stats, loss_stats = train(model_instance, args)
            # Explicitly free the memory of the model
            del model_instance
            gc.collect()
            # Synchronize with code in cuda
            if torch.cuda.is_available():
                torch.cuda.synchronize()


            # Save the accuracy and loss statistics
            saveMetrics(accuracy_stats, file_name + "_accuracies", results_path, args)
            saveMetrics(loss_stats, file_name + "_losses", results_path, args)
            i += 1


def train(model, args):
    """
    Function for training the given model with the specified parameters
    Args:
        model: model to train
        args: training parameters

    Returns:
        accuracy_stats(dict): Statistics of the accuracy metric for validation and training set
        loss_stats(dict): Statistics of the loss metric for validation and training set
    """
    # Get argument for verbose (printing extra information)
    verbose: bool = args["verbose"]

    # Define the device to run the script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"[GENERAL] Using {device}")

    # Get parameters defined by args
    model_feature_extractor: str = args["model"]
    modality_feature_extractor: str = args["modality"]
    shift: str = args["shift"]
    batch_size: int = args["batch_size"]
    learning_rate: float = args["learning_rate"]

    # Instantiate the dataset and the data loader and the model.

    dataset_train = FeaturesDataset(model=model_feature_extractor,
                                    modality=modality_feature_extractor,
                                    shift=shift, train=True)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)

    dataset_test = FeaturesDataset(model=model_feature_extractor,
                                   modality=modality_feature_extractor,
                                   shift=shift, train=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)

    # Use the cross entropy loss and the Adam optimizer

    weight_decay = args["weight_decay"]
    early = args["early"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Do the training

    # This is the number of times we go through the entire data!
    epochs: int = args["epochs"]
    if verbose:
        print("[GENERAL] The model used is:\n")
        print(model)

    # Initialize metrics about training
    accuracy_stats: dict = {
        'train': [],
        "val": []
    }
    loss_stats: dict = {
        'train': [],
        "val": []
    }

    if verbose:
        print("[GENERAL] Starting training")
        print("[GENERAL] Moving model to GPU")
    model.to(device)

    last_loss = np.inf
    trigger_times: int = 0
    patience: int = 10
    for e in tqdm(range(1, epochs + 1)):
        # Set up model for training
        model.train()

        train_epoch_loss = 0
        train_epoch_acc = 0

        # Iterate trough batches
        for X_train_batch, y_train_batch in dataloader_train:
            # Move data to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            # Set the gradient to 0
            optimizer.zero_grad()
            # Apply model to the batch. Forward propagation
            # Check if the model need transposition of the input
            if args["transpose_input"]:
                X_train_batch = torch.transpose(X_train_batch, -2, -1)

            y_train_pred = model(X_train_batch)

            batch_size = X_train_batch.size()[0]

            # Resize for the cross entropy loss. todo: add here constant num_classes
            y_train_pred = torch.reshape(y_train_pred, (batch_size, 8))
            y_train_batch = torch.reshape(y_train_batch, (batch_size,))
            # Compute loss
            train_loss = criterion(y_train_pred, y_train_batch)
            train_accuracy = multiclassAccuracy(y_train_pred, y_train_batch)

            # Backpropagate the gradient. Take a step in "correct" direction
            train_loss.backward()  # Accumulates gradients
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_accuracy.item()

            # Now do validation
            with torch.no_grad():
                # Stop keeping track of gradients
                val_epoch_loss = 0
                val_epoch_acc = 0
                model.eval()
                for X_val_batch, y_val_batch in dataloader_test:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    if args["transpose_input"]:
                        X_val_batch = torch.transpose(X_val_batch, -2, -1)

                    y_val_pred = model(X_val_batch)

                    batch_size = X_val_batch.size()[0]

                    # Resize for the cross entropy loss. todo: add here constant num_classes
                    y_val_pred = torch.reshape(y_val_pred, (batch_size, 8))
                    y_val_batch = torch.reshape(y_val_batch, (batch_size,))
                    # Compute loss
                    val_loss = criterion(y_val_pred, y_val_batch)

                    val_accuracy = multiclassAccuracy(y_val_pred, y_val_batch)
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_accuracy.item()

        if early:
            current_loss = val_epoch_loss
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else:
                # Reset trigger value
                trigger_times = 0

            last_loss = current_loss

        # Append the average batch loss
        loss_stats["train"].append(train_epoch_loss / len(dataloader_train))
        accuracy_stats["train"].append(train_epoch_acc / len(dataloader_train))
        # Append the average batch accuracy
        loss_stats["val"].append(val_epoch_loss / len(dataloader_test))
        accuracy_stats["val"].append(val_epoch_acc / len(dataloader_test))

        # print message every certain epoch
        if e % (epochs // args["frequency_validation"]) == 0:
            tqdm.write(
                f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(dataloader_train):.5f}\
                | Train Acc: {train_epoch_acc / len(dataloader_train):.3f}\
                | Val Acc : {val_epoch_acc / len(dataloader_test)}    ')

    return accuracy_stats, loss_stats


def instantiateModels(args):
    r""" Function instantiating the models to be tested with their respective parameters (for eventual grid search).

    Args:
        args : dict: Arguments of the program. It has to contain the key "temporal_aggregator" that specifies the temporal
            aggregation method used.

    Returns:

    """
    aggregator: str = args["temporal_aggregator"]
    # Dictionary with available models for temporal aggregation
    models = {"AvgPooling": NeuralNetAvgPooling,
              "TRN": TRNmodule.RelationModuleMultiScaleWithClassifier,
              "MultiAttention": MultiHeadAttentionClassifier,
              "MultiBaseLine" : AttentionBaseline,
              "MultiAttentionPytorch" : MultiHeadAttentionClassifierPyTorch}
    # Dictionary with the parameters to test in each model (internal and external)

    avg_pooling_parameters = {"dropout": [0, 0.25, 0.5], "hidden_sizes": [[512], [512, 64]]}
    avg_pooling_external = {"early": [True, False], "weight_decay": [0, 1e-5, 1e-6]}

    trn_parameters = {"dropout": [0.6, 0.1, 0.2, 0.7], "img_feature_dim": [2048], "num_frames": [5], "num_class": [8]}
    trn_external = {"early": [True, False], "weight_decay": [0, 1e-5, 1e-6]}

    multi_parameters = {"attention_heads": [1,2], "dropout": [0.1,0.5],"hidden_sizes": [[512],[512,1024]],"encoder_layers" : [1,2],
                        "key_size" : [512,1024,2048]}
    multi_external = {"early": [False], "weight_decay": [1e-6,0, 1e-5]}

    multi_done_parameters = {"d_model":[2048], "d_inner":[512], "n_head" : [3], "d_k" : [512], "d_v":[512]}
    multi_done_external = multi_external

    multi_baseline_parameters = {"dropout" : [0.5] ,"number_heads" : [3],"hidden_sizes" : [[1024,512]]}
    multi_baseline_external = multi_external

    multi_pytorch_parameters = {"attention_heads": [2,2], "dropout": [0.8, 0.5], "hidden_sizes": [[512, 1024],[512], [512, 1024]],
                        "encoder_layers": [1, 2],
                        "key_size": [512, 1024, 2048]}
    multi_pytorch_external = {"early": [False], "weight_decay": [1e-6, 0, 1e-5]}
    # Define dictionaries
    parameters = {"AvgPooling": avg_pooling_parameters,
                  "TRN": trn_parameters,
                  "MultiAttention": multi_parameters,
                  "MultiBaseLine" : multi_baseline_parameters,
                  "MultiAttentionPytorch" : multi_pytorch_parameters}

    external = {"AvgPooling": avg_pooling_external,
                "TRN": trn_external,
                "MultiAttention":multi_external,
                "MultiBaseLine" : multi_baseline_external,
                "MultiAttentionPytorch" : multi_pytorch_external}

    return models[aggregator], parameters[aggregator], external[aggregator]


if __name__ == '__main__':
    main()