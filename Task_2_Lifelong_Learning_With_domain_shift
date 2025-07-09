# It implements a custom Convolutional Neural Network (CNN) trained from scratch
# using a "Learning without Prejudice" (LwP) style continual learning strategy
# enhanced with Experience Replay to handle the domain shifts present in
# datasets D11 through D20.
#
# Key Features:
# 1. No Pre-trained Model: A custom CNN is defined and trained from scratch.
# 2. Experience Replay: A memory buffer stores samples from past tasks to mitigate
#    catastrophic forgetting when the data distribution changes.


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
INITIAL_EPOCHS = 20
UPDATE_EPOCHS = 10
TEMPERATURE = 2.0
MEMORY_SIZE_PER_TASK = 100

# --- 1. End-to-End Continual CNN Model (from Scratch) ---
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

# --- 2. Experience Replay Buffer ---
class ExperienceReplayBuffer:
    def __init__(self, memory_size_per_task):
        self.memory_size_per_task = memory_size_per_task
        self.buffer = []

    def add(self, dataset):
        num_samples = len(dataset)
        indices = random.sample(range(num_samples), min(num_samples, self.memory_size_per_task))
        images = dataset.tensors[0][indices]
        labels = dataset.tensors[1][indices]
        self.buffer.append((images, labels))

    def get_data(self):
        if not self.buffer:
            return None
        all_images = torch.cat([images for images, labels in self.buffer], dim=0)
        all_labels = torch.cat([labels for images, labels in self.buffer], dim=0)
        return torch.utils.data.TensorDataset(all_images, all_labels)

# --- 3. Data Loading and Formatting Utilities ---
def load_data(task_folder, task_id):
    """Loads the train and test data for a specific task."""
    train_path = os.path.join(task_folder, f'D{task_id}_train.pt')
    test_path = os.path.join(task_folder, f'D{task_id}_test.pt')
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    train_dataset = torch.utils.data.TensorDataset(train_data['images'], train_data['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_data['images'], test_data['labels'])
    return train_dataset, test_dataset

def format_matrix_string(matrix):
    """Formats a list of lists into a readable string."""
    s = ""
    for row in matrix:
        s += '[ ' + ' '.join([f'{val:6.2f}' for val in row]) + ' ]\n'
    return s

# --- 4. Knowledge Distillation Loss ---
def distillation_loss(y, teacher_scores, T):
    """Computes the KL divergence between student and teacher logits."""
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / T, dim=1),
                                             F.softmax(teacher_scores / T, dim=1)) * (T * T)

# --- 5. Training and Evaluation Functions ---
def evaluate(model, test_loader):
    """Evaluates the model's accuracy on a given test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_initial(model, train_loader, optimizer, criterion, epochs):
    """Trains the model on the first, fully labeled dataset (D1)."""
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

def train_update(model, old_model, train_loader, optimizer, criterion, replay_loader):
    """Updates the model on a new, unlabeled task using LwP and Experience Replay."""
    model.train()
    old_model.eval()
    for epoch in range(UPDATE_EPOCHS):
        replay_iter = iter(replay_loader) if replay_loader else None
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

            if replay_iter:
                try:
                    replay_images, replay_labels = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_images, replay_labels = next(replay_iter)
                replay_images, replay_labels = replay_images.to(DEVICE), replay_labels.to(DEVICE)
                replay_outputs = model(replay_images)
                loss_replay = criterion(replay_outputs, replay_labels)
                total_loss += loss_replay

            total_loss.backward()
            optimizer.step()
        print(f"Update Training: Epoch [{epoch+1}/{UPDATE_EPOCHS}], Loss: {total_loss.item():.4f}")

# --- 6. Main Execution ---
def run_task2_experiment():
    """Runs the full lifelong learning experiment for Task 2."""
    print(f"\n{'='*20} RUNNING TASK 2 (WITH DOMAIN SHIFT) {'='*20}")
    
    task_folder = 'data/task2_domain_shift'
    start_id = 11
    end_id = 20
    num_tasks = end_id - start_id + 1
    
    model = ContinualCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    replay_buffer = ExperienceReplayBuffer(MEMORY_SIZE_PER_TASK)
    all_test_loaders = []
    
    print("\n--- Training on Initial Task D1 ---")
    d1_train_data, d1_test_data = load_data('data/task1_no_domain_shift', 1)
    d1_train_loader = torch.utils.data.DataLoader(d1_train_data, batch_size=BATCH_SIZE, shuffle=True)
    d1_test_loader = torch.utils.data.DataLoader(d1_test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_test_loaders.append(d1_test_loader)
    
    train_initial(model, d1_train_loader, optimizer, criterion, INITIAL_EPOCHS)
    replay_buffer.add(d1_train_data)
    
    accuracy_matrix = [[0.0 for _ in range(num_tasks + 1)] for _ in range(num_tasks)]
    
    # --- Sequential Training Loop ---
    for i in range(num_tasks):
        task_id = start_id + i
        print(f"\n--- Training on Task {task_id} ---")
        train_data, test_data = load_data(task_folder, task_id)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        all_test_loaders.append(test_loader)
        
        old_model = ContinualCNN().to(DEVICE)
        old_model.load_state_dict(model.state_dict())
        
        replay_dataset = replay_buffer.get_data()
        replay_loader = torch.utils.data.DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        train_update(model, old_model, train_loader, optimizer, criterion, replay_loader)
        replay_buffer.add(train_data)

        print(f"--- Evaluating model after training on Task {task_id} ---")
        # Evaluate on D1
        acc_d1 = evaluate(model, all_test_loaders[0])
        accuracy_matrix[i][0] = acc_d1
        print(f"  Accuracy on Task D1 test set: {acc_d1:.2f}%")
        # Evaluate on D11 through current task
        for j in range(i + 1):
            eval_task_id = start_id + j
            acc = evaluate(model, all_test_loaders[j+1])
            accuracy_matrix[i][j+1] = acc
            print(f"  Accuracy on Task {eval_task_id} test set: {acc:.2f}%")
            
    return accuracy_matrix

if __name__ == '__main__':
    accuracy_matrix_2 = run_task2_experiment()

    print("\n\n" + "="*50)
    print("FINAL RESULTS FOR TASK 2 (FROM SCRATCH)")
    print("="*50)
    print("\n--- Accuracy Matrix for Task 2 (With Domain Shift & Experience Replay) ---")
    print("Rows: Model after Task i, Cols: Accuracy on Task j test set")
    print(format_matrix_string(accuracy_matrix_2))

