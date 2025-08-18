# Key Features:
# 1. No Pre-trained Model: A custom CNN is defined and trained from scratch.
# 2. End-to-End Training: The entire network is updated during each learning phase.
# 3. Minimized Dependencies: Uses only the `torch` library and standard Python modules.


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# --- Configuration ---
# This section defines the key hyperparameters for the experiment.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64               # The number of samples processed in one forward/backward pass.
LEARNING_RATE = 1e-4          # The step size for the Adam optimizer.
INITIAL_EPOCHS = 20           # Number of epochs to train on the first, labeled dataset (D1).
UPDATE_EPOCHS = 10            # Number of epochs for updating the model on subsequent (unlabeled) tasks.
TEMPERATURE = 2.0             # A parameter for knowledge distillation. A higher temperature softens the
                              # probability distribution from the teacher model, transferring more
                              # nuanced information to the student model.

# --- 1. End-to-End Continual CNN Model (from Scratch) ---
# This class defines the entire neural network architecture. It's composed of two main parts:
# a feature extractor (convolutional layers) and a classifier (fully-connected layers).
class ContinualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ContinualCNN, self).__init__()
        # The 'features' part is the convolutional backbone of the network.
        # It's designed to learn hierarchical visual features from the raw pixel data.
        self.features = nn.Sequential(
            # Block 1: Learns simple features like edges and colors.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (RGB), 32 output feature maps.
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # Downsamples the feature maps by half.

            # Block 2: Learns more complex features by combining features from Block 1.
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Takes 32 feature maps, outputs 64.
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Learns even more abstract features.
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Takes 64 feature maps, outputs 128.
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # The 'classifier' part is the head of the network. It takes the learned features
        # and makes the final decision about which class the image belongs to.
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                            # Regularization to prevent overfitting.
            nn.Linear(128 * 4 * 4, 512),                # Takes the flattened feature vector and transforms it.
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)                 # The final output layer with 10 neurons, one for each class.
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.features(x)         # 1. Pass the image through the feature extractor.
        x = x.view(x.size(0), -1)    # 2. Flatten the 3D feature maps into a 1D vector.
        x = self.classifier(x)       # 3. Pass the feature vector through the classifier.
        return x

# --- 2. Data Loading and Formatting Utilities ---
def load_data(task_folder, task_id):
    """Loads the train and test data for a specific task from .pt files."""
    train_path = os.path.join(task_folder, f'D{task_id}_train.pt')
    test_path = os.path.join(task_folder, f'D{task_id}_test.pt')
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    train_dataset = torch.utils.data.TensorDataset(train_data['images'], train_data['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_data['images'], test_data['labels'])
    return train_dataset, test_dataset

def format_matrix_string(matrix):
    """Formats a list of lists (the accuracy matrix) into a nicely readable string."""
    s = ""
    for row in matrix:
        s += '[ ' + ' '.join([f'{val:6.2f}' for val in row]) + ' ]\n'
    return s

# --- 3. Knowledge Distillation Loss ---
def distillation_loss(y, teacher_scores, T):
    """
    Computes the knowledge distillation loss. This is the core of the LwP strategy.
    It encourages the student model (y) to mimic the output distribution of the
    teacher model (teacher_scores), not just its final prediction.
    """
    # We use KL Divergence loss between the softened outputs of the student and teacher.
    # Softening is done by dividing logits by the temperature (T).
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / T, dim=1),
                                             F.softmax(teacher_scores / T, dim=1)) * (T * T)

# --- 4. Training and Evaluation Functions ---
def evaluate(model, test_loader):
    """Evaluates the model's accuracy on a given test set."""
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.).
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for efficiency.
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max logit as the prediction.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_initial(model, train_loader, optimizer, criterion, epochs):
    """Trains the model on the first, fully labeled dataset (D1). This is standard supervised training."""
    model.train() # Set the model to training mode.
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()      # Clear previous gradients.
            outputs = model(images)    # Forward pass.
            loss = criterion(outputs, labels) # Calculate standard cross-entropy loss.
            loss.backward()            # Backward pass to compute gradients.
            optimizer.step()           # Update model weights.
        print(f"Initial Training: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def train_update(model, old_model, train_loader, optimizer, criterion):
    """Updates the model on a new, unlabeled task using the LwP strategy."""
    model.train()    # The current model is the "student" and is in training mode.
    old_model.eval() # The old model is the "teacher" and is in evaluation mode.
    for epoch in range(UPDATE_EPOCHS):
        for images, _ in train_loader: # We ignore the true labels of the new task.
            images = images.to(DEVICE)
            # Use the teacher model to generate labels for the new data.
            with torch.no_grad():
                teacher_scores = old_model(images)
                # "Hard" labels for the classification loss part.
                pseudo_labels = torch.argmax(teacher_scores, dim=1)

            optimizer.zero_grad()
            # The student model makes its own predictions on the new data.
            outputs = model(images)
            
            # The loss is a combination of two parts:
            # 1. Distillation Loss: Encourages the student to mimic the teacher's full output distribution.
            loss_distill = distillation_loss(outputs, teacher_scores, TEMPERATURE)
            # 2. Pseudo-Label Loss: A standard classification loss using the teacher's predictions as ground truth.
            loss_pseudo = criterion(outputs, pseudo_labels)
            
            total_loss = loss_distill + loss_pseudo
            total_loss.backward()
            optimizer.step()
        print(f"Update Training: Epoch [{epoch+1}/{UPDATE_EPOCHS}], Loss: {total_loss.item():.4f}")

# --- 5. Main Execution ---
def run_task1_experiment():
    """Runs the full lifelong learning experiment for Task 1."""
    print(f"\n{'='*20} RUNNING TASK 1 (NO DOMAIN SHIFT) {'='*20}")
    
    task_folder = 'data/task1_no_domain_shift'
    num_tasks = 10
    
    # Initialize the model, optimizer, and loss function.
    model = ContinualCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Store all test loaders for final evaluation at each step.
    all_test_loaders = []
    
    # --- Initial Training on D1 ---
    print("\n--- Training on Initial Task D1 ---")
    d1_train_data, d1_test_data = load_data(task_folder, 1)
    d1_train_loader = torch.utils.data.DataLoader(d1_train_data, batch_size=BATCH_SIZE, shuffle=True)
    d1_test_loader = torch.utils.data.DataLoader(d1_test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_test_loaders.append(d1_test_loader)
    
    train_initial(model, d1_train_loader, optimizer, criterion, INITIAL_EPOCHS)
    
    # --- Setup Accuracy Matrix ---
    # Rows: Model after training on task i. Cols: Accuracy on test set of task j.
    accuracy_matrix = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    acc = evaluate(model, d1_test_loader)
    print(f"Accuracy on D1 test set after initial training: {acc:.2f}%")
    accuracy_matrix[0][0] = acc
    
    # --- Sequential Training Loop for Tasks D2 to D10 ---
    for i in range(1, num_tasks):
        task_id = i + 1
        print(f"\n--- Training on Task {task_id} ---")
        train_data, test_data = load_data(task_folder, task_id)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        all_test_loaders.append(test_loader)
        
        # Create a copy of the model to act as the "teacher".
        old_model = ContinualCNN().to(DEVICE)
        old_model.load_state_dict(model.state_dict())
        
        # Update the main model (the "student") using the teacher's knowledge.
        train_update(model, old_model, train_loader, optimizer, criterion)
        
        # Evaluate the updated model on all tasks seen so far.
        print(f"--- Evaluating model after training on Task {task_id} ---")
        for j in range(i + 1):
            eval_task_id = j + 1
            acc = evaluate(model, all_test_loaders[j])
            accuracy_matrix[i][j] = acc
            print(f"  Accuracy on Task {eval_task_id} test set: {acc:.2f}%")
            
    return accuracy_matrix

if __name__ == '__main__':
    # Run the experiment for Task 1.
    accuracy_matrix_1 = run_task1_experiment()

    # Print the final results.
    print("\n\n" + "="*50)
    print("FINAL RESULTS FOR TASK 1 (FROM SCRATCH)")
    print("="*50)
    print("\n--- Accuracy Matrix for Task 1 (No Domain Shift) ---")
    print("Rows: Model after Task i, Cols: Accuracy on Task j test set")
    print(format_matrix_string(accuracy_matrix_1))
