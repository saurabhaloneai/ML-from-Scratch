import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)


outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = XORModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10000
for epoch in range(epochs):
   
    outputs_pred = model(inputs)
    
    # loss
    loss = criterion(outputs_pred, outputs)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

   
    if epoch % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    predictions = model(inputs).round()


print("Predictions:")
print(predictions.numpy())
