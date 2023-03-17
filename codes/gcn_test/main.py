from torch_geometric.loader import DataLoader
from train import *

if __name__ == "__main__":

    # choose the TUDataset or MNIST,
    # or another graph classification problem if preferred
    dataset = TUDataset(root="./tmp", name="PROTEINS")
    #dataset = GNNBenchmarkDataset(root="./tmp", name="MNIST")

    # shuffle dataset and get train/validation/test splits
    dataset = dataset.shuffle()

    num_samples = len(dataset)
    batch_size = 32

    num_val = num_samples // 10

    val_dataset = dataset[:num_val]
    test_dataset = dataset[num_val:2 * num_val]
    train_dataset = dataset[2 * num_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for batch in train_loader:
        break

    num_features = batch.x.shape[1]
    num_classes = dataset.num_classes

    plain_model = PlainGCN(num_features=num_features, num_classes=num_classes)
    lr = 1e-5
    num_epochs = 2500

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(plain_model.parameters(), lr=lr)

    train_model(plain_model, train_loader, criterion, optimizer, num_epochs=num_epochs, verbose=True, val_loader=val_loader)