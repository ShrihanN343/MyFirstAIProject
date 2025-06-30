# designed for sequential data, LSTM = Long Short Term Memory
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        # referencing parent constructor function from nn.Module (parent)
        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear=nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # reshape input to (bash, seq_len, input_size)
        # x=x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # returns output and tuple of hidden cell states 
        last_time_step_out=lstm_out[:,-1,:]
        out=self.linear(last_time_step_out)
        return out

class LSTMTrainer:
    # self, for any class, is an argument that references the object (instance of a class)
    # epoch = number of training cycles
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1, epochs=20, batch_size=32, learning_rate=0.001, random_state=None):
        self.model=LSTMRegressor(input_size, hidden_size, num_layers, output_size)
        self.epochs=epochs
        self.batch_size=batch_size
        self.criterion=nn.MSELoss() # mean squared error (MSE) loss function, measures how well model predictions match true target (measures error, inaccuracy)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate) # optimizer adjusts model weights based on the criterion/loss function
    def fit(self, x, y):
        # convert pandas  to torch tensors (matrix w/ n number of dimensions), our challenge in ML is ensuring our dimensions align from layer to layer in a way that maximizes model accuracy
        x_tensor=torch.tensor(x.values, dtype=torch.float32).unsqueeze(-1) # adds feature dimension
        y_tensor=torch.tensor(y.values, dtype=torch.float32).unsqueeze(-1)
    
        dataset=TensorDataset(x_tensor,y_tensor)
        loader=DataLoader(dataset,batch_size=self.batch_size,shuffle=True)

        for epoch in range(self.epochs):
            for features,labels in loader:
                self.optimizer.zero_grad()
                outputs=self.model(features)
                loss=self.criterion(outputs,labels)
                loss.backward() # calculates loss for each weight by doing back propogation
                self.optimizer.step()
            # print(f"epoch {epoch}, loss: {loss.items():.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor=torch.tensor(x.values,dtype=torch.float32).unsqueeze(-1)
            predictions = self.model(x_tensor)
        return predictions.numpy().flatten()
