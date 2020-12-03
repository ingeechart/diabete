from src.model import ASPModel
from src.dataset import *
from src.utils import Normalization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import sys
import os

_SEQUENCE_LENGTH = 96*3
_INPUT_DIM = 50
_HIDDEN_DIM = 50
_EMBEDDING_DIM = 50
_LEARNING_RATE = 0.001
_EPOCHS = 2600
_BATCH_SIZE = 128
_CUDA_FLAG = torch.cuda.is_available()

_MODEL_LOAD_FLAG = False
_MODEL_PATH = "data\\models"
_MODEL_LOAD_NAME = "ASPModel_{}_checkpoint.pth"

def train():
    # Load objects for training
    train_dataset = ASPDataset(mode = "train")
    train_dataloader = DataLoader(train_dataset, batch_size = _BATCH_SIZE, shuffle = True)
    test_dataset = ASPDataset(mode = "test")
    test_dataloader = DataLoader(test_dataset, batch_size = _BATCH_SIZE, shuffle = False)

    # Model load
    model = ASPModel(seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM)
    if _MODEL_LOAD_FLAG :
        model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME)))
    if _CUDA_FLAG : model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = _LEARNING_RATE)
    norm = Normalization()
    writer = SummaryWriter("Tensorboard")

    for cur_epoch in range(_EPOCHS):
        # Learning Rate Scheduler
        if cur_epoch == 300 :
            optimizer.param_groups[0]["lr"] = 0.0001
        elif cur_epoch == 500 :
            optimizer.param_groups[0]["lr"] = 0.00001
            
        # Training
        model.train()
        optimizer.zero_grad()
        train_total_loss = 0.0
        for cur_iter, train_data in enumerate(train_dataloader):
            # Data load
            train_inputs, train_labels = train_data
            if _CUDA_FLAG :
                train_inputs = train_inputs.cuda()
                train_labels = train_labels.cuda()
            _, temp_length = train_inputs.shape

            # Update parameters
            train_outputs = model(train_inputs).view(-1, temp_length)
            train_labels = norm.normalize(train_labels)       # Experimental
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            train_total_loss += train_loss.detach()

            if cur_epoch % 600 == 599 : 
                _test_sample("train_prediction", norm.de_normalize(train_labels), norm.de_normalize(train_outputs), train_inputs)
                break

            print("TRAIN ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, train_total_loss/len(train_dataloader)))

        # Evaludation
        model.eval()
        with torch.no_grad() :
            test_total_loss = 0.0
            for cur_iter, test_data in enumerate(test_dataloader):
                # Data load
                test_inputs, test_labels = test_data
                if _CUDA_FLAG :
                    test_inputs = test_inputs.cuda()
                    test_labels = test_labels.cuda()
                _, temp_length = test_inputs.shape

                test_outputs = model(test_inputs).view(-1, temp_length)
                test_total_loss += criterion(test_outputs, norm.normalize(test_labels))

                if cur_epoch % 600 == 599 : 
                    _test_sample("test_prediction", test_labels, norm.de_normalize(test_outputs), test_inputs)
                    break

            print("TEST ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, test_total_loss/len(test_dataloader)))

        writer.add_scalars("Loss", {"train_loss" : train_total_loss/len(train_dataloader), "test_loss" : test_total_loss/len(test_dataloader)}, cur_epoch)
        if cur_epoch% 600 == 599 :  
            torch.save(model.state_dict(), os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME.format(cur_epoch)))
            break
    writer.close()

def _test_sample(name, label, output, input):
    tempWriter = SummaryWriter("{}".format(name))
    input_data = input[0]
    prediction = output[0]
    label_data = label[0]
    predict = torch.cat((input_data, prediction), dim = 0)
    true = torch.cat((input_data, label_data), dim = 0)
    for i in range(288*2):
        tempWriter.add_scalars("Glucose", {"True" : true[i], "prediction" : predict[i]}, i)
    tempWriter.close()

if __name__ == "__main__":
    train()
