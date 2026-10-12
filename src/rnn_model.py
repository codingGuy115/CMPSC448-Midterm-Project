
# --------------------------------------------------
# Step 15: Prepare data for the RNN (LSTM)
# --------------------------------------------------

import torch
import torch.nn as nn
import copy

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

# Use the same random seed as our CNN
torch.manual_seed(42)

# Convert token sequences into PyTorch tensors
X_train = torch.tensor(X_train_output, dtype=torch.long)
X_val = torch.tensor(X_val_output, dtype=torch.long)
X_test = torch.tensor(X_test_output, dtype=torch.long)

# Convert LLM labels into tensors
Y_train = torch.tensor(y_train, dtype=torch.long)
Y_val = torch.tensor(y_val, dtype=torch.long)
Y_test = torch.tensor(y_test, dtype=torch.long)

print("\n=== RNN Data Preparation ===")
print("Training inputs:", X_train.shape)
print("Validation inputs:", X_val.shape)
print("Testing inputs:", X_test.shape)
print("Training labels:", Y_train.shape)
print("Vocabulary size:", len(vocab))


# --------------------------------------------------
# Step 16: Define the LSTM architecture
# --------------------------------------------------

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128,
                 hidden_dim=64, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Count real tokens before padding
        lengths = (x != 0).sum(dim=1).cpu()

        # Convert token IDs into embeddings
        x = self.embedding(x)

        # Ignore padded positions in the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Process the sequence through the LSTM
        _, (hidden, _) = self.lstm(packed)

        # Use the final hidden state
        x = hidden[-1]

        x = self.dropout(x)

        # Produce GPT, Claude, and Gemini scores
        x = self.fc(x)

        return x


# --------------------------------------------------
# Step 17: LSTM Training Setup
# --------------------------------------------------

from torch.utils.data import TensorDataset, DataLoader

# Create the LSTM model
model = TextLSTM(vocab_size=len(vocab))

# Pair responses with their correct LLM labels
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# Organize the data into batches
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

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\n=== LSTM Training Setup ===")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Loss function:", criterion.__class__.__name__)
print("Optimizer:", optimizer.__class__.__name__)


# --------------------------------------------------
# Step 18: Train and validate the LSTM
# --------------------------------------------------

NUM_EPOCHS = 10
best_val_loss = float("inf")
best_model_state = None
best_epoch = 0

for epoch in range(NUM_EPOCHS):

    # TRAINING
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

    # VALIDATION
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}",
        flush=True
    )


# Restore the model with the lowest validation loss
model.load_state_dict(best_model_state)

print("\n=== Best LSTM Model ===")
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")

# --------------------------------------------------
# Step 19: Evaluate LSTM on the test dataset
# --------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix

model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = test_outputs.argmax(dim=1)

test_accuracy = (
    (test_predictions == Y_test).sum().item()
    / len(Y_test)
)

print("\n=== LSTM Test Results ===")
print(f"Test Accuracy: {test_accuracy:.2%}")
print("Correct predictions:",
      (test_predictions == Y_test).sum().item())
print("Total test responses:", len(Y_test))

class_names = ["GPT", "Claude", "Gemini"]

actual_labels = Y_test.tolist()
predicted_labels = test_predictions.tolist()

print("\n=== LSTM Classification Report ===")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)

print("\n=== LSTM Confusion Matrix ===")
print(
    confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2]
    )
)
