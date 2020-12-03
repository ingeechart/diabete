import torch
SEQ_LEN = 4
HIDDEN_DIM = 1
EMBEDDING_DIM = 7
INPUT_DIM = 1

class custom_LSTM(torch.nn.Module):
    def __init__(self, seq_len = SEQ_LEN, hidden_dim = HIDDEN_DIM, input_dim = INPUT_DIM, embedding_dim = EMBEDDING_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.embedding_dim = embedding_dim

        #self.embedding = torch.nn.Embedding(self.input_dim, self.embedding_dim)
        self.embedding = Embedding(input_dim = self.input_dim)
        self.lstm = torch.nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim)
        self.linear = torch.nn.Linear(in_features = self.hidden_dim, out_features = self.input_dim)

    def forward(self, x):
        input_vectors = self.embedding(x)
        input_vectors = input_vectors.view(len(x), -1, self.input_dim).contiguous()
        output, _ = self.lstm(input_vectors)
        output = self.linear(output)
        return output
        
class Normalization(object):
    def __init__(self, min_dim = 0, max_dim = 12):
        super().__init__()
        self.xmin = torch.tensor(min_dim, dtype = torch.float32)
        self.xmax = torch.tensor(max_dim, dtype = torch.float32)

    def __call__(self, x):
        norm_x = (x - self.xmin)/(self.xmax - self.xmin)
        return norm_x

class Embedding(object):
    def __init__(self, min_dim = 0, max_dim = 12, batch_num = 1, input_dim= 1):
        super().__init__()
        self.norm = Normalization(min_dim, max_dim)
        self.batch_num = batch_num
        self.input_dim = input_dim
    
    def __call__(self, x):
        output_list = []
        for data in x:
            data = torch.tensor(self.norm(data), dtype = torch.float32, requires_grad = True).view(1, 1).contiguous()
            output_list.append(data)
        
        for i in range(1, len(x)):
            output_list[0] = torch.cat((output_list[0], output_list[i]), dim = 0)
        output = output_list[0]
        output = output.view(len(x), self.batch_num, self.input_dim)
        return output

training_data = [
                torch.tensor([0, 1, 2, 3], dtype = torch.float32, requires_grad =False),\
                torch.tensor([1, 2, 3, 4], dtype = torch.float32, requires_grad =False),\
                torch.tensor([2, 3, 4, 5], dtype = torch.float32, requires_grad =False),\
                torch.tensor([3, 4, 5, 6], dtype = torch.float32, requires_grad =False)\
                ]
"""
target = [
            # 0, 1, 2, 3, 4, 5, 6, 7
            torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False)\
        ]
label_data = [] 
label_data.append(torch.cat((target[1], target[2], target[3], target[4])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[2], target[3], target[4], target[5])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[3], target[4], target[5], target[6])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[4], target[5], target[6], target[0])).view(4, 1, 7).contiguous())
"""
target = [
            # 0, 1, 2, 3, 4, 5, 6, 7
            torch.tensor([0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([1], dtype = torch.float32, requires_grad = False),\
            torch.tensor([2], dtype = torch.float32, requires_grad = False),\
            torch.tensor([3], dtype = torch.float32, requires_grad = False),\
            torch.tensor([4], dtype = torch.float32, requires_grad = False),\
            torch.tensor([5], dtype = torch.float32, requires_grad = False),\
            torch.tensor([6], dtype = torch.float32, requires_grad = False),\
            torch.tensor([7], dtype = torch.float32, requires_grad = False)\
        ]
label_data = [] 
label_data.append(torch.cat((target[3], target[4], target[5], target[6])).view(4, 1, 1).contiguous())
label_data.append(torch.cat((target[4], target[5], target[6], target[0])).view(4, 1, 1).contiguous())
label_data.append(torch.cat((target[5], target[6], target[0], target[1])).view(4, 1, 1).contiguous())
label_data.append(torch.cat((target[6], target[0], target[1], target[2])).view(4, 1, 1).contiguous())

class train(object):
    def __init__(self, epochs = 2500, lr = 0.01):
        self.epochs = epochs
        self.lr = lr
        
    def __call__(self, x):
        lstm = custom_LSTM(seq_len = SEQ_LEN, hidden_dim = HIDDEN_DIM, input_dim = INPUT_DIM, embedding_dim = EMBEDDING_DIM)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(lstm.parameters(), lr = self.lr)
        lstm.train()        
        for j in range(self.epochs):
            for i in range(len(training_data)):
                output = lstm(training_data[i])
                

                loss = criterion(output, label_data[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
        
        lstm.eval()
        output = lstm(torch.tensor([6, 6, 6, 6], dtype = torch.float32))
        print(output)

if __name__ == "__main__":
    training = train()
    training(training_data, )