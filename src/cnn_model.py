import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from preprocessing import prepare_experiment

# Choose which RQ2 experiment to run
MODE = "combined"  # Options: "input", "output", "combined"

(
    X_train_output,
    X_val_output,
    X_test_output,
    y_train,
    y_val,
    y_test,
    vocab
) = prepare_experiment(MODE)

# --------------------------------------------------
# Step 9: Convert prepared data into PyTorch tensors
# --------------------------------------------------

X_train = torch.tensor(X_train_output, dtype=torch.long)
X_val = torch.tensor(X_val_output, dtype=torch.long)
X_test = torch.tensor(X_test_output, dtype=torch.long)

Y_train = torch.tensor(y_train, dtype=torch.long)
Y_val = torch.tensor(y_val, dtype=torch.long)
Y_test = torch.tensor(y_test, dtype=torch.long)

print("\n=== CNN Data Preparation ===")
print("Training inputs:", X_train.shape)
print("Validation inputs:", X_val.shape)
print("Testing inputs:", X_test.shape)

print("Training labels:", Y_train.shape)
print("Vocabulary size:", len(vocab))


# --------------------------------------------------
# Step 10: Define the CNN architecture
# --------------------------------------------------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128,
                 num_filters=64, kernel_size=3,
                 num_classes=3):
        super().__init__()

        # Convert token IDs into word embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Detect patterns across neighboring tokens
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size
        )

        # Reduce each filter's output to one value
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Help prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Predict one of three LLM classes
        self.fc = nn.Linear(num_filters, num_classes)

    

    def forward(self, x):
        # Save original token IDs to identify padding
        token_ids = x

        # Convert token IDs into embeddings
        x = self.embedding(x)

        # Rearrange dimensions for Conv1d
        x = x.permute(0, 2, 1)

        # Apply convolution and activation
        x = torch.relu(self.conv(x))

        # Calculate the number of real tokens
        valid_lengths = (token_ids != 0).sum(dim=1)

        # Calculate valid convolution positions
        conv_lengths = valid_lengths - self.conv.kernel_size[0] + 1

        # Create a mask to exclude padded positions
        positions = torch.arange(
            x.size(2), device=x.device
        ).unsqueeze(0)

        mask = positions < conv_lengths.unsqueeze(1)

        # Exclude padded positions from pooling
        x = x.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # Pool the strongest features
        x = self.pool(x).squeeze(2)

        # Apply dropout
        x = self.dropout(x)

        # Produce three classification scores
        x = self.fc(x)

        return x


# --------------------------------------------------
# Step 11: CNN Training Setup
# --------------------------------------------------

from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducible results
torch.manual_seed(42)

# Create a fresh CNN model
model = TextCNN(vocab_size=len(vocab))

# Combine training inputs with their correct labels
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# Divide the datasets into smaller batches
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# Define how we measure mistakes
criterion = nn.CrossEntropyLoss()

# Define how the model updates its parameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\n=== CNN Training Setup ===")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Loss function:", criterion.__class__.__name__)
print("Optimizer:", optimizer.__class__.__name__)


# --------------------------------------------------
# Step 12: Train and validate the CNN
# --------------------------------------------------

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()

    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )


# --------------------------------------------------
# Step 13: Evaluate CNN on the test dataset
# --------------------------------------------------

model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = test_outputs.argmax(dim=1)

test_accuracy = (
    (test_predictions == Y_test).sum().item()
    / len(Y_test)
)

print("\n=== CNN Test Results ===")
print(f"Test Accuracy: {test_accuracy:.2%}")
print("Correct predictions:",
      (test_predictions == Y_test).sum().item())
print("Total test responses:", len(Y_test))


# --------------------------------------------------
# Step 14: Detailed CNN Evaluation
# --------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix

class_names = ["GPT", "Claude", "Gemini"]

actual_labels = Y_test.tolist()
predicted_labels = test_predictions.tolist()

print("\n=== CNN Classification Report ===")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)

print("\n=== CNN Confusion Matrix ===")
print(
    confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2]
    )
)
