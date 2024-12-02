import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

batch_size = 64
learning_rate = 0.001
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

image, label = train_dataset[10]

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  
        self.dropout1 = nn.Dropout(0.5) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5) 
        self.fc3 = nn.Linear(128, 10) 

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)      
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)         
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)         
        x = self.fc3(x)
        return x

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(torch.device('cuda') if torch.cuda.is_available() else 'cpu'), \
                             labels.to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        test_model(model, test_loader)

def test_model(model, test_loader, is_output = False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(torch.device('cuda') if torch.cuda.is_available() else 'cpu'), \
                             labels.to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if is_output:
                for idx2 in range(len(images)):
                    data['ID'].append((idx * batch_size) + idx2 + 1)
                    data['target'].append(predicted[idx2].item())
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data = {
    "ID": [],
    "target": [],
}

train_model(model, train_loader, criterion, optimizer, num_epochs)
test_model(model, test_loader, True)

df = pd.DataFrame(data)
df.to_csv("csvs/data.csv", index=False)
