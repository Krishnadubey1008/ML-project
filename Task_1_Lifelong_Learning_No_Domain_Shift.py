import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
INITIAL_EPOCHS = 20
UPDATE_EPOCHS = 10
TEMPERATURE = 2.0

class ContinualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ContinualCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_data(task_folder, task_id):
    train_path = os.path.join(task_folder, f'D{task_id}_train.pt')
    test_path = os.path.join(task_folder, f'D{task_id}_test.pt')
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    train_dataset = torch.utils.data.TensorDataset(train_data['images'], train_data['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_data['images'], test_data['labels'])
    return train_dataset, test_dataset

def format_matrix_string(matrix):
    s = ""
    for row in matrix:
        s += '[ ' + ' '.join([f'{val:6.2f}' for val in row]) + ' ]\n'
    return s

def distillation_loss(y, teacher_scores, T):
    return nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(y / T, dim=1),
        F.softmax(teacher_scores / T, dim=1)
    ) * (T * T)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_initial(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Initial Training: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def train_update(model, old_model, train_loader, optimizer, criterion):
    model.train()
    old_model.eval()
    for epoch in range(UPDATE_EPOCHS):
        for images, _ in train_loader:
            images = images.to(DEVICE)
            with torch.no_grad():
                teacher_scores = old_model(images)
                pseudo_labels = torch.argmax(teacher_scores, dim=1)

            optimizer.zero_grad()
            outputs = model(images)
            loss_distill = distillation_loss(outputs, teacher_scores, TEMPERATURE)
            loss_pseudo = criterion(outputs, pseudo_labels)
            total_loss = loss_distill + loss_pseudo
            total_loss.backward()
            optimizer.step()
        print(f"Update Training: Epoch [{epoch+1}/{UPDATE_EPOCHS}], Loss: {total_loss.item():.4f}")

def run_task1_experiment():
    print(f"\n{'='*20} RUNNING TASK 1 (NO DOMAIN SHIFT) {'='*20}")
    task_folder = 'data/task1_no_domain_shift'
    num_tasks = 10
    
    model = ContinualCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    all_test_loaders = []
    
    print("\n--- Training on Initial Task D1 ---")
    d1_train_data, d1_test_data = load_data(task_folder, 1)
    d1_train_loader = torch.utils.data.DataLoader(d1_train_data, batch_size=BATCH_SIZE, shuffle=True)
    d1_test_loader = torch.utils.data.DataLoader(d1_test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_test_loaders.append(d1_test_loader)
    
    train_initial(model, d1_train_loader, optimizer, criterion, INITIAL_EPOCHS)

    accuracy_matrix = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    acc = evaluate(model, d1_test_loader)
    print(f"Accuracy on D1 test set after initial training: {acc:.2f}%")
    accuracy_matrix[0][0] = acc
    
    for i in range(1, num_tasks):
        task_id = i + 1
        print(f"\n--- Training on Task {task_id} ---")
        train_data, test_data = load_data(task_folder, task_id)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        all_test_loaders.append(test_loader)
        
        old_model = ContinualCNN().to(DEVICE)
        old_model.load_state_dict(model.state_dict())
        
        train_update(model, old_model, train_loader, optimizer, criterion)
        
        print(f"--- Evaluating model after training on Task {task_id} ---")
        for j in range(i + 1):
            acc = evaluate(model, all_test_loaders[j])
            accuracy_matrix[i][j] = acc
            print(f"  Accuracy on Task {j+1} test set: {acc:.2f}%")
            
    return accuracy_matrix

if __name__ == '__main__':
    accuracy_matrix_1 = run_task1_experiment()
    print("\n\n" + "="*50)
    print("FINAL RESULTS FOR TASK 1 (FROM SCRATCH)")
    print("="*50)
    print("\n--- Accuracy Matrix for Task 1 (No Domain Shift) ---")
    print("Rows: Model after Task i, Cols: Accuracy on Task j test set")
    print(format_matrix_string(accuracy_matrix_1))
