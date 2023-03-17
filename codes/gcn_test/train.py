from model import *

def evaluate(model, test_loader, save_results=True, tag="_default", verbose=False):

    # get test accuracy score

    num_correct = 0.
    num_total = 0.

    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0
    total_batches = 0

    for batch in test_loader:

        pred = model(batch.to(my_device))

        loss = criterion(pred, batch.y.to(my_device))

        num_correct += (pred.argmax(dim=1) == batch.y).sum()
        num_total += pred.shape[0]

        total_loss += loss.detach()
        total_batches += batch.batch.max()

    test_loss = total_loss / total_batches
    test_accuracy = num_correct / num_total

    if verbose:
        print(f"accuracy = {test_accuracy:.4f}")

    results = {
        "accuracy": test_accuracy,
        "loss": test_loss,
        "tag": tag
    }

    return results

def train_model(model, train_loader, criterion, optimizer, num_epochs=1000,
                verbose=True, val_loader=None, save_tag="default_run_"):

    ## call validation function and print progress at each epoch end
    display_every = 1 #num_epochs // 10
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(my_device)

    # we'll log progress to tensorboard
    log_dir = f"lightning_logs/plain_model_{str(int(time.time()))[-8:]}/"
    writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    for epoch in range(num_epochs):

        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad()

            pred = model(batch.to(my_device))
            loss = criterion(pred, batch.y.to(my_device))
            loss.backward()

            optimizer.step()

            total_loss += loss.detach()
            batch_count += 1

        mean_loss = total_loss / batch_count

        writer.add_scalar("loss/train", mean_loss, epoch)

        if epoch % display_every == 0:
            train_results = evaluate(model, train_loader, tag=f"train_ckpt_{epoch}_", verbose=False)
            train_loss = train_results["loss"]
            train_accuracy = train_results["accuracy"]

            if verbose:
                print(f"training loss & accuracy at epoch {epoch} = "f"{train_loss:.4f} & {train_accuracy:.4f}")

            if val_loader is not None:
                val_results = evaluate(model, val_loader, tag=f"val_ckpt_{epoch}_", verbose=False)
                val_loss = val_results["loss"]
                val_accuracy = val_results["accuracy"]

                if verbose:
                    print(f"val. loss & accuracy at epoch {epoch} = "f"{val_loss:.4f} & {val_accuracy:.4f}")
                else:
                    val_loss = float("Inf")
                    val_acc = - float("Inf")

                writer.add_scalar("loss/train_eval", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("accuracy/train", train_accuracy, epoch)
                writer.add_scalar("accuracy/val", val_accuracy, epoch)