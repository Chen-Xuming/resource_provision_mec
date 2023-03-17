from load_data import PulpSolutionDataset
from model_fc1 import GCN4OPTIMAL
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import time

def train_model(model, train_loader, criterion, optimizer, num_epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = "./log/" + date_time
    writer = SummaryWriter(log_dir)

    model_file = "./model_checkpoint/" + date_time + ".pth"

    lr_decay_factor = 0.94

    step = 0
    for epoch in range(num_epoch):
        batch_loss = []
        iterator = tqdm(train_loader, unit='Batch')

        for i, batch in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, num_epoch))

            optimizer.zero_grad()

            output = model.forward(batch.to(device))
            output = torch.squeeze(output, dim=1)         # fc1模型

            loss = criterion(output, batch.y.to(device))
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            step += 1

            iterator.set_postfix({'loss': '{%.6f}' % loss.item()})
            writer.add_scalar("loss", loss.item(), step)

        # iterator.set_postfix({'batch_loss': '{%.6f}' % np.average(batch_loss)})

        if epoch % 200 == 0:
            torch.save(model.state_dict(), model_file)
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay_factor


if __name__ == '__main__':
    num_feature = 5
    embedding_size = 128
    batch_size = 128
    learning_rate = 1e-5
    num_epoch = 3000

    dataset = PulpSolutionDataset()
    dataset = dataset.shuffle()
    train_loader = DataLoader(dataset, batch_size=batch_size)

    # model = GCN4OPTIMAL(num_feature=num_feature, embedding_size=embedding_size, num_class=dataset.get_num_class())
    # criterion = torch.nn.CrossEntropyLoss()

    model = GCN4OPTIMAL(num_feature=num_feature, embedding_size=embedding_size)
    criterion = torch.nn.MSELoss(reduction="sum")


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model=model, train_loader=train_loader, criterion=criterion,
                optimizer=optimizer, num_epoch=num_epoch)

