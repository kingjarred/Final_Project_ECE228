'''
=======================================
Recurring Neural Network Preconditioner
=======================================
''' 

import torch
import torch.nn as nn

# define the rnn approximator class
class rnnApprox(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(rnnApprox, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# function to train the rnn model
def train_rnn(model, criterion, optimizer, x_train, y_train, num_epochs=1000):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:  # print every 100 epochs
            print(f'epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')
    return losses

# function to approximate a given function using the trained model
def approximate_function(x, model):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        return model(x_tensor).item()
