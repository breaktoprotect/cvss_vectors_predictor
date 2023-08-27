import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from training_data import X_train, cia_output
import time


# ? Setup and Configuration
# Hyper params
NUM_OF_EPOCHS = 15
BATCH_SIZE = 8
TEXT_MAX_LENGTH = 256
LEARNING_RATE = 1e-5


def main():
    # Display metadata
    print(f"[*] Welcome to text classification multi-class model training!")
    print(f"[$] Hyper Parameters:")
    print(f"    Num of Epochs: {NUM_OF_EPOCHS}")
    print(f"    Batch Size: {BATCH_SIZE}")
    print(f"    TEXT_MAX_LENGTH: {TEXT_MAX_LENGTH}")
    print(f"    LEARNING_RATE: {LEARNING_RATE}")
    print(f"[*] Training initializing...")

    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Available device for training: -> {device}")

    # Load pre-trained transformer model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )
    model.to(device)

    # Load outputs to convert to y_train
    y_train = []
    for data in cia_output:
        y_train.append(data["confidentiality"])

    print(f"[*] Total number of training records: {len(y_train)}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_descriptions = tokenizer(
        X_train,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=TEXT_MAX_LENGTH,
    )

    label_map = {"none": 0, "low": 1, "high": 2}
    labels = [label_map[label] for label in y_train]

    dataset = TensorDataset(
        tokenized_descriptions["input_ids"],
        tokenized_descriptions["attention_mask"],
        torch.tensor(labels),
    )
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # * Commence Training
    train_model(
        model, NUM_OF_EPOCHS, train_dataset, train_dataloader, device, LEARNING_RATE
    )

    # Move model back to CPU before saving (if needed)
    model.to("cpu")

    # Save the trained model
    model.save_pretrained("confidentiality_model")
    tokenizer.save_pretrained("confidentiality_model")


def train_model(
    model,
    NUM_OF_EPOCHS,
    train_dataset,
    train_dataloader,
    hardware_device,
    learning_rate,
):
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_predictions = 0
    total_start_time = time.time()

    model.train()

    # Training loop
    for epoch in range(NUM_OF_EPOCHS):
        total_loss = 0
        correct_predictions = 0
        epoch_start_time = time.time()

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids, attention_mask, target = batch
            input_ids = input_ids.to(hardware_device)  # Move input tensors to GPU
            attention_mask = attention_mask.to(
                hardware_device
            )  # Move input tensors to GPU
            target = target.to(hardware_device)  # Move target tensor to GPU

            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate the number of correct predictions
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == target).item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct_predictions / len(train_dataset)
        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time

        print(
            f"[*] Epoch {epoch + 1}/{NUM_OF_EPOCHS}, Training loss: {avg_train_loss:.4f} | Training accuracy: {train_accuracy:.4f} | Time elapsed: {epoch_elapsed_time:.2f} sec"
        )

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"[*] Total elapsed time: {total_elapsed_time:.2f} sec")
    print(
        f"[+] Training completed with total Epoch: {epoch + 1} - Total loss: {loss:.4f} - Elapsed Time: {total_elapsed_time:.2f}"
    )


if __name__ == "__main__":
    main()
