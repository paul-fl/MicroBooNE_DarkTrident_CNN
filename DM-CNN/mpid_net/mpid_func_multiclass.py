import torch
import torch.nn as nn
import torch.nn.functional as F

def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        print("Model training status in train_step,", model.training)
        y_prediction = model(x)
        loss = loss_fn(y_prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

def make_test_step(model, test_loader, loss_fn, optimizer):
    def test_step(test_loader, train_device):
        model.eval()
        print("Model training status in test_step,", model.training)
        tot_loss = 0
        ctr = 0
        for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(test_loader):
            x_batch = x_batch.to(train_device).view((-1, 1, 512, 512))
            y_batch = y_batch.to(train_device)
            y_prediction = model(x_batch)
            loss = loss_fn(y_prediction, y_batch)
            tot_loss += loss.item()
            ctr += 1
            if ctr == 10:
                break
        return tot_loss / float(ctr)
    return test_step

def validation(model, test_loader, batch_size, device, event_nums):
    model.eval()
    predicted = 0.0
    total = 0.0
    for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(test_loader):
        if batch_idx * batch_size > event_nums:
            break
        x_batch = x_batch.to(device).view((-1, 1, 512, 512))
        y_batch = y_batch.to(device)
        outputs = model(x_batch)
        _, predicted_classes = torch.max(outputs, 1)
        correct = (predicted_classes == y_batch).sum().item()
        predicted += correct
        total += y_batch.size(0)
    return float(predicted) / total

