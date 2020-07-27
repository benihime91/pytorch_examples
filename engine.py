import numpy as np
import torch
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn import metrics
from torch import optim
from tqdm.notebook import tqdm
from recorder import Recoder

TEMPLATE = "Epoch {} - train loss: {:.3f}  val loss: {:.3f}  accuracy: {:.3f}%  error_rate: {:.4f}"


def calc_metrics(logits, yb):
    """calculates accuracy and error rate"""
    _, preds = torch.max(logits, 1)
    # calculate accuracy
    acc = metrics.accuracy_score(yb.cpu().numpy(), preds.cpu().numpy())
    # calculate error rate
    error_rate = 1 - acc
    return acc, error_rate


def train(model, dataloader, optimizer, criterion, scheduler, device, mb, recorder):
    """training step"""
    loss = 0.
    model.to(device)
    optimizer.zero_grad()
    model.train()

    # Iterate over the DataLoader
    for n, (xb, yb) in enumerate(progress_bar(dataloader, parent=mb)):
        xb, yb = xb.to(device), yb.to(device)
        # Forward pass
        logits = model(xb)
        loss_value = criterion(logits, yb)
        loss += loss_value.item()
        mb.child.comment = f"batch_loss: {loss_value.item():.3f}"

        try:
            recorder.update_lr_moms(
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["momentum"])

        except:
            recorder.update_lr_moms(optimizer.param_groups[0]["lr"], 0)

        # BackWard Pass
        loss_value.backward()
        torch.nn.utils.clip_grad_value_(
            [p for p in model.parameters() if p.requires_grad], 0.1)

        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    # Average Loss for the epoch
    loss /= len(dataloader)
    return loss


def validate(model, dataloader, criterion, device, mb=None):
    """validaiton_step"""
    loss, acc, error_rate = 0., 0., 0.
    model.eval()
    model.to(device)
    # Iterate over the dataloader batches
    for n, (xb, yb) in enumerate(progress_bar(dataloader, parent=mb)):
        with torch.no_grad():
            xb, yb = xb.to(device), yb.to(device)
            # Forward pass
            logits = model(xb)
            loss += criterion(logits, yb).item()
            acc_n, error_rate_n = calc_metrics(logits, yb)
            acc += acc_n
            error_rate += error_rate_n
    # Average Statistics for the Epoch
    loss /= len(dataloader)
    acc /= len(dataloader)
    error_rate /= len(dataloader)
    return loss, acc, error_rate


def save_model(model, best_loss, loss, path):
    """Saves model whenever validation loss decreases"""
    if loss < best_loss:
        torch.save(model.state_dict(), path)
    return loss


class Engine:
    def __init__(self):
        self.recorder = Recoder()

    def fit_one_cycle(self, dataloaders, model, optimizer, criterion, epochs, max_lr, device="cpu", path=None):
        """
        Implements OneCyclePolicy
        arguments:
            1. dataloaders : train and validation dataloaders as dict
                            {"train":train_dataloader,"validation":val_dataloader}
            2. model : user defined model
            3. optimizer : uder defined optimizer
            4. criterion : user defined loss function
            5. epochs : no. of epochs to train for
            6. max_lr : maximum learning rate the scheduler
                        can reach during one cycle training
            7. device : device on which to train [one of ("cpu" or "cuda:0")]
                        [default: "cpu"]
        """
        best_loss = np.Inf
        train_dl, val_dl = dataloaders["train"], dataloaders["validation"]
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)
        mb = master_bar(range(epochs))
        for n in mb:
            # Training Step
            train_loss = train(model, train_dl, optimizer, criterion,
                               scheduler, device, mb, recorder=self.recorder)
            # Validaiton Step
            val_loss, val_acc, error_rate = validate(
                model, val_dl, criterion, device, mb)
            # Update Recoder
            self.recorder.update_loss_metrics(
                train_loss, val_loss, val_acc, error_rate)
            if path is not None:
                best_loss = save_model(model, best_loss, val_loss, path)
            mb.write(TEMPLATE.format(n, train_loss,
                                     val_loss, val_acc*100., error_rate))

        # Load best Model
        model.load_state_dict(torch.load(path))

    @staticmethod
    def evaluate(data_loader, model, criterion, device="cpu"):
        """evaluates the model"""
        loss, acc = validate(model, data_loader, criterion, device)
        return loss, acc  # Return loss and accuracy

    @staticmethod
    def predict(data_loader, model, class_names: dict = None, device="cpu"):
        """Predict classes for the given DataLoader"""
        model.to(device)
        model.eval()
        final_predicitons = []
        with torch.no_grad():
            for n, data in enumerate(tqdm(data_loader)):
                data = data[0]
                data = data.to(device)
                with torch.no_grad():
                    preds = model(data)
                _, predictions = torch.max(preds, 1)
                predictions = predictions.cpu().numpy()

                for i in predictions:
                    if class_names is not None:
                        final_predicitons.append(class_names[i])
                    else:
                        final_predicitons.append(i)

        return final_predicitons
