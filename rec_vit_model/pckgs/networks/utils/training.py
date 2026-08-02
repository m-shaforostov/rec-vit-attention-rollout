import torch
import torchvision.transforms.functional as F
import wandb
import random
import numpy as np
from tqdm import tqdm
import torchvision


def train_and_test_CNN(model, criterion, optimizer, scheduler, device, dataset_sizes, test_loader, train_loader=None,
                   num_epochs=150, train=True, wandb_log=False):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        modes = ['train', 'test'] if train else ['test']
        for phase in modes:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loader = train_loader if phase == 'train' else test_loader
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'test' and wandb_log:
                wandb.log({"loss": epoch_loss, "accuracy": epoch_acc})

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print()

    return model


class CustomOptimizer:
    # Inspired by https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
    def __init__(self, optimizer, default_lr=0.01, lr_decay=0.95, warmup=(False, 0), lr_multiplier=1):
        self.optimizer = optimizer
        self.use_warmup, self.warmup_steps = warmup
        self.current_step = 0
        self.default_lr = default_lr
        self.lr_decay = lr_decay
        self.lr_multiplier = lr_multiplier

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.current_step += 1
        if self.use_warmup and self.warmup_steps >= self.current_step:
            lr = self.current_step/self.warmup_steps * self.default_lr
        else:
            lr = self.lr_decay ** (self.current_step - self.warmup_steps) * self.default_lr
        lr *= self.lr_multiplier

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def test_rvit(model, test_loader, n_loops, device, random_tokens=False, specific_token=None, return_predictions=False,
              blur=False, inv_blur=False, random_transform=False, random_blur=0, specific_blur=0, blur_value=(7, 4)):
    model.eval()

    if return_predictions:
        predictions = torch.zeros((len(test_loader.dataset), n_loops))
    curr_idx = 0
    accuracy = [0] * n_loops
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = data.to(device), target.to(device)

        # onehot_decode if the target is one-hot encoded
        if len(target.shape) == 2:
            target = torch.argmax(target, dim=1)

        current_batch_size = data.shape[0]

        # the class token needs to be 'repeated' here (due to multiple forward passes at once - batch_size !=1)
        if random_tokens:
            if specific_token is not None:
                cls_token = specific_token.expand(current_batch_size, -1, -1)
            else:
                embed_dim = model.cls_token.shape[2]
                cls_token = torch.randn(current_batch_size, 1, embed_dim).to(device) * torch.std(model.cls_token)
        else:
            cls_token = model.cls_token.expand(current_batch_size, -1, -1)

        with torch.no_grad():
            if random_transform:
                rot = np.random.rand() * 20 - 10
                trans = (np.random.randint(-5, 6), np.random.randint(-5, 6))
                scale = np.random.rand() * 0.2 + 0.9

            if blur:
                blr = torchvision.transforms.GaussianBlur(kernel_size=blur_value[0], sigma=blur_value[1])
                blurred_data = [data]
                for i in range(n_loops - 1):
                    blurred_data.append(blr(blurred_data[-1]))
                if inv_blur:
                    blurred_data = blurred_data[::-1]

            # blur of images during testing, levels of blur are set by random_blur. The prediction is averaged
            if specific_blur != 0:
                blr = torchvision.transforms.GaussianBlur(kernel_size=blur_value[0], sigma=blur_value[1])
                for _ in range(specific_blur):
                    data = blr(data)

            if random_blur != 0:
                blur_preds = [0] * (random_blur+1)
                blr = torchvision.transforms.GaussianBlur(kernel_size=blur_value[0], sigma=blur_value[1])

                for rb in range(random_blur+1):
                    output, cls_token = model(data, cls_token)
                    blur_preds[rb] = output.detach().cpu().numpy()
                    data = blr(data)

                b_pred = torch.Tensor(np.sum(np.array(blur_preds), axis=0)).to(device)
                pred = b_pred.data.max(1, keepdim=True)[1]

                accuracy[0] += pred.eq(target.data.view_as(pred)).sum()

            else:
                for i in range(n_loops):
                    if blur:
                        data = blurred_data[i]

                    if random_transform:
                        data = F.affine(data, rot, trans, scale, 0)

                    output, cls_token = model(data, cls_token)
                    pred = output.data.max(1, keepdim=True)[1]
                    if return_predictions:
                        predictions[curr_idx:curr_idx+current_batch_size, i] = torch.squeeze(pred)
                    accuracy[i] += pred.eq(target.data.view_as(pred)).sum()

        curr_idx += current_batch_size

    for i in range(n_loops):
        accuracy[i] = (100. * accuracy[i] / len(test_loader.dataset)).cpu().numpy()

    return (accuracy, predictions) if return_predictions else accuracy


def test_rvit_classes(model, test_loader, n_loops, device, random_tokens=False, specific_token=None,
                      return_predictions=False):
    model.eval()
    # number of classes is fixed=10
    if return_predictions:
        predictions = torch.zeros((len(test_loader.dataset), n_loops))
    curr_idx = 0
    accuracy = np.zeros((10, n_loops))
    normalisation = np.zeros(10)
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = data.to(device), target.to(device)
        current_batch_size = data.shape[0]

        # the class token needs to be 'repeated' here (due to multiple forward passes at once - batch_size !=1)
        if random_tokens:
            if specific_token is not None:
                cls_token = specific_token.expand(current_batch_size, -1, -1)
            else:
                embed_dim = model.cls_token.shape[2]
                cls_token = torch.randn(current_batch_size, 1, embed_dim).to(device) * torch.std(model.cls_token)
        else:
            cls_token = model.cls_token.expand(current_batch_size, -1, -1)

        with torch.no_grad():
            for i in range(n_loops):
                output, cls_token = model(data, cls_token)
                pred = output.data.max(1, keepdim=True)[1]
                if return_predictions:
                    predictions[curr_idx:curr_idx+current_batch_size, i] = torch.squeeze(pred)

                for j in range(10):  # number of classes is fixed=10
                    mask = target == j
                    trg = target[mask]
                    prd = torch.squeeze(pred[mask])
                    accuracy[j, i] += torch.sum(trg == prd)
                    if i == 0:
                        normalisation[j] += torch.sum(mask)

        curr_idx += current_batch_size

    for j in range(10):
        accuracy[j, :] /= normalisation[j]

    return (accuracy, predictions) if return_predictions else accuracy


def train_rvit(model, train_loader, train_option, device, scheduler, criterion, n_loops, log_interval, ep,
               cls_regularizer=False, cls_reg_alpha=.1, turn_off_input=False, random_target=False, blur=False,
               random_transform=False, inv_blur=False, random_blur=0, blur_value=(7, 4)):
    model.train()

    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        d_shape = data.shape
        current_batch_size = data.shape[0]
        cls_token = model.cls_token.expand(current_batch_size, -1, -1)

        scheduler.zero_grad()
        loss = 0

        cls_toks = [cls_token]

        if random_transform:
            rot = np.random.rand() * 20 - 10
            trans = (np.random.randint(-5, 6), np.random.randint(-5, 6))
            scale = np.random.rand() * 0.2 + 0.9

        if blur:
            blr = torchvision.transforms.GaussianBlur(kernel_size=blur_value[0], sigma=blur_value[1])
            blurred_data = [data]
            for i in range(n_loops - 1):
                blurred_data.append(blr(blurred_data[-1]))
            if inv_blur:
                blurred_data = blurred_data[::-1]

        if random_blur != 0:
            blr = torchvision.transforms.GaussianBlur(kernel_size=blur_value[0], sigma=blur_value[1])
            for blur_idx in range(random.randrange(0, random_blur+1)):
                data = blr(data)

        for i in range(n_loops):
            if blur:
                data = blurred_data[i]

            if random_transform:
                data = F.affine(data, rot, trans, scale, 0)

            if turn_off_input:
                r = np.random.rand()
                if r <= i/10:
                    output, cls_token = model(torch.zeros(d_shape).to(device), cls_token)
                else:
                    output, cls_token = model(data, cls_token)
            else:
                output, cls_token = model(data, cls_token)
            cls_toks.append(cls_token)
            if train_option == 2:
                if random_target and np.random.rand() > (i+3)/10 and i != n_loops-1:
                    fal_targ = np.random.choice(torch.max(target).cpu(), target.shape[0])
                    fal_targ = torch.LongTensor(fal_targ).to(device)

                    loss += criterion(output, fal_targ) / n_loops
                else:
                    loss += criterion(output, target) / n_loops

        if train_option == 1:
            loss = criterion(output, target)

        if cls_regularizer:
            reg_term = 0
            for i in range(0, len(cls_toks)-1):
                for j in range(i+1, len(cls_toks)):
                    m1 = cls_toks[i]
                    m2 = cls_toks[j]
                    multi = torch.squeeze(torch.bmm(m1, torch.reshape(m2, (m1.shape[0], m1.shape[2], m1.shape[1]))))
                    norms = torch.linalg.norm(torch.squeeze(m1), dim=1) * torch.linalg.norm(torch.squeeze(m2), dim=1)
                    reg_term = torch.sum(torch.abs(torch.squeeze(multi)) / norms)
            loss += cls_reg_alpha * reg_term

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % log_interval == log_interval - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return total_loss / len(train_loader.dataset)


def train_and_test_rvit(model, train_loader, test_loader, optimizer, device, scheduler=None, epochs=10,
                        name="model_weights", save=False, criterion=None, wandb_log=False, n_loops=3,
                        train_option=1, log=None, cls_regularizer=False, cls_reg_alpha=.1, turn_off_input=False,
                        random_t=False, random_transform=False, blur=False, blurinv=False, random_blur=False,
                        blur_value=(7, 4)):

    model.to(device)

    for epoch in range(epochs):
        scheduler.update_learning_rate()

        train_loss = train_rvit(
            model=model,
            train_loader=train_loader,
            train_option=train_option,
            device=device,
            scheduler=scheduler,
            criterion=criterion,
            n_loops=n_loops,
            log_interval=50,
            ep=epoch,
            cls_regularizer=cls_regularizer,
            cls_reg_alpha=cls_reg_alpha,
            turn_off_input=turn_off_input,
            random_target=random_t,
            random_transform=random_transform,
            blur=blur,
            inv_blur=blurinv,
            random_blur=random_blur,
            blur_value=blur_value,
        )

        accuracy = test_rvit(
            model=model,
            test_loader=test_loader,
            n_loops=n_loops,
            device=device,
            blur=blur,
            inv_blur=blurinv,
            random_transform=random_transform,
            random_blur=random_blur,
            blur_value=blur_value,
        )

        if wandb_log:
            lg = {"loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
            for i in range(len(accuracy)):
                lg[f'acc{i+1}'] = accuracy[i]
            lg['avg_acc'] = np.mean(np.array(accuracy))
            wandb.log(lg)
        if log is not None:
            for i in range(len(accuracy)):
                log[0][log[1]-1, log[2], i, epoch] = accuracy[i]

    if save:
        torch.save(model.state_dict(), f'{name}.pth')

