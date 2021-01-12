from torchvision import datasets, transforms
from models import mnistmodels as models
from modules import base

import os
import datetime
import shutil
import torch
import random
random.seed(2)
torch.manual_seed(2)
half_precision = False
cuda_on = True
if cuda_on:
    cuda_device = torch.device("cuda")
else:
    cuda_device = torch.device("cpu")

precision = torch.float16 if half_precision else torch.float32

class Trainer():

    def __init__(self, my_model, **kwargs):
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(os.getcwd(), "log/" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        for f in ["mnisttest.py", "models/mnistmodels.py"]:
            src = os.path.join(os.getcwd(), f)
            dst = os.path.join(self.log_dir, f)
            folder = "/".join(dst.split("/")[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            shutil.copyfile(src, dst)
        self.model = my_model
        self.noise_scale = kwargs.get("noise_scale", 3e-3)
        # self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 1e-3)
        self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 0)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay",0)
        # self.lr_decay = kwargs.get("lr_decay", 1e-2)
        self.ave_delta_rate = kwargs.get("ave_delta_rate", .99)
        self.epochs = kwargs.get("epochs", 1000)
        self.batches_per_epoch = kwargs.get("batches_per_epoch",1)
        self.batch_size = kwargs.get("batch_size",1)
        self.directions = kwargs.get("directions",1)
        self.max_steps = kwargs.get("max_steps", -1)

    def train(self):
        self.model.batch_size=self.batch_size
        if half_precision:
            self.model = self.model.half()
        perturbed_model = base.PerturbedModel(self.model, self.directions)
        ave_delta = .005 * self.batch_size
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay, eps=1e-3)
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=True, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.1307,), (0.3081,))
        #                    ])),
        #     drop_last = True,
        #     batch_size=self.batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            drop_last = True,
            batch_size=self.batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])),
        #     batch_size=self.test_batch_size, shuffle=True)

        for epoch in range(self.epochs):
            print("Epoch:",epoch)
            total_reward = 0.0
            total_game_length = 0.0
            total_cards = 0.0
            total_points = 0.0
            max_cards = 0
            max_score = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.cuda()
                # data = (data * 255).type(torch.int8)
                target = target.cuda()
                with torch.no_grad():
                    perturbed_model.set_seed()
                    perturbed_model.set_noise_scale(self.noise_scale)
                    perturbed_model.allocate_memory()
                    perturbed_model.set_noise()
                    pred = perturbed_model.forward(data)
                    # reward = -torch.nn.NLLLoss(reduce=False)(pred,target)
                    reward = torch.nn.CrossEntropyLoss(reduction="none")(pred, target)
                    result = reward - reward.mean()
                    # step_size = result / ((ave_delta + 1e-5) * self.noise_scale)
                    step_size = result
                    # ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * (result.norm(p=1))
                    perturbed_model.update(step_size)
                    total_reward += reward.mean()
                print(reward.mean())
                # (torch.nn.NLLLoss()(self.model.forward(data),target)).backward()

                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())
                self.noise_scale *= self.noise_scale_decay
                opt.step()
                # print(torch.nn.NLLLoss()(self.model.forward(data),target))
            # for param in self.model.parameters():
            #
            #     print(param.data.abs().mean())
            print("Average Reward:", total_reward / self.batches_per_epoch)
            # print("Average Game Length:", total_game_length.float() / (self.batches_per_epoch * self.batch_size))
            fname = os.path.join(self.checkpoints_dir, "epoch_"+str(epoch)+".pkl")
            perturbed_model.free_memory()
            torch.save(self.model, fname)





if __name__ == "__main__":
    batch_size = 2 ** 8
    directions = 2 ** 8

    # my_model = models.MNISTConvNet(directions=directions, action_size=10,in_channels=1)
    my_model = models.MNISTDenseNet(directions=directions, action_size=10,in_channels=1)
    # my_model = models.MNISTBinaryDenseNet(directions=directions, action_size=10,in_channels=1)
    # Trainer(model.TransformerNet()).train()
    if cuda_on:
        my_model = my_model.cuda()
    Trainer(my_model,batch_size=batch_size,directions=directions).train()



