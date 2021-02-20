from torchvision import datasets, transforms
from models import mnistmodels as models
from modules import base
# from modules import binary

from extensions import booleanOperations
import os
import datetime
import shutil
import torch
import random
random.seed(2)
torch.manual_seed(3)
device = torch.device("cuda")
from timing import timer
import gc
def get_memory():
    count = 0
    total_size = 0
    total_size_param = 0
    total_size_tensor = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj), obj.numel())
                count += 1
                total_size += obj.numel()
                if isinstance(obj, torch.nn.parameter.Parameter):
                    total_size_param += obj.numel()
                else:
                    total_size_tensor += obj.numel()
        except:
            pass
    print(count, total_size, total_size_tensor, total_size_param)

class LabelSmoothLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        return loss





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
        for f in ["cifar10test.py", "models/cifar10models.py"]:
            src = os.path.join(os.getcwd(), f)
            dst = os.path.join(self.log_dir, f)
            folder = "/".join(dst.split("/")[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            shutil.copyfile(src, dst)
        self.model = my_model
        # self.noise_scale = kwargs.get("noise_scale", 3e-4)
        # self.noise_scale = kwargs.get("noise_scale", 3e-3)
        self.noise_scale = kwargs.get("noise_scale", 2e-3)
        self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 1e-3)
        # self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 0)
        # self.lr = kwargs.get("lr", 3e-4)
        self.lr = kwargs.get("lr", 1e-6)
        self.lr_decay = 1 - kwargs.get("lr_decay", 1e-4)
        self.lr_scale = 1.0
        self.weight_decay = kwargs.get("weight_decay", 2e-4)
        self.ave_delta_rate = kwargs.get("ave_delta_rate", .99)
        self.epochs = kwargs.get("epochs", 1000)
        self.batch_size = kwargs.get("batch_size", 1)
        self.directions = kwargs.get("directions", 1)
        self.max_steps = kwargs.get("max_steps", -1)
        # self.label_smoothing = kwargs.get("label_smoothing", 0.01)
        self.label_smoothing = kwargs.get("label_smoothing", 0.00)


    def train(self):
        self.model.batch_size = self.batch_size
        perturbed_model = base.PerturbedModel(self.model, self.directions)
        ave_delta = .005 * self.batch_size
        # opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay, eps=1e-3)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0)

        # train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=True, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.1307,), (0.3081,))
        #                    ])),
        #     drop_last = True,
        #     batch_size=self.batch_size, shuffle=True)
        transform = transforms.Compose([
            transforms.ToTensor()])
        self.dataset = datasets.MNIST('../data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                        ]))
        self.images = torch.FloatTensor(self.dataset.data.float()).unsqueeze(1).to(device)
        self.labels = self.dataset.targets
        self.labels_tensor = torch.LongTensor(self.labels).to(device)
        self.loss = LabelSmoothLoss(self.label_smoothing)
        train_set_size = self.images.shape[0]
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            drop_last = True,
            batch_size=self.batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])),
        #     batch_size=self.test_batch_size, shuffle=True)

        for epoch in range(self.epochs):
            print("Epoch:", epoch)
            total_reward = 0.0
            total_accuracy = 0.0
            batch_count = train_set_size // self.batch_size
            for i in range(batch_count):
                idx = torch.randint(train_set_size, (self.batch_size,))
                data = self.images[idx]
                target = self.labels_tensor[idx]
                # data = data.cuda()
                # data = (data * 255).type(torch.int8)
                # data = booleanOperations.int8pack(data.permute([0, 2, 3, 1]).contiguous().view(-1, 3), dtype=torch.int32).view(-1, 32, 32, 1)
                # target = target.cuda()

                with torch.no_grad():
                    perturbed_model.set_seed()
                    perturbed_model.set_noise_scale(self.noise_scale)
                    perturbed_model.allocate_memory()
                    perturbed_model.set_noise()
                    import time
                    pred = perturbed_model.forward(data)
                    # get_memory()
                    # continue
                    # print(pred)
                    # reward = torch.nn.NLLLoss(reduce=False)(pred, target)
                    reward = torch.nn.CrossEntropyLoss(reduction="none")(pred, target)
                    # reward = self.loss.forward(pred, target)
                    # print(reward.shape)
                    _, pred_label = torch.max(pred, dim=1)
                    accuracy = (pred_label == target).sum().float() / pred.shape[0]
                    result = reward - reward.mean()
                    # step_size = result / ((ave_delta + 1e-5) * self.noise_scale)
                    step_size = result * self.lr_scale / self.noise_scale
                    # ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * (result.norm(p=1))
                    perturbed_model.update(step_size)
                    total_reward += reward.mean()
                    total_accuracy += accuracy
                print("accuracy", accuracy, "reward", reward.mean())
                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())
                self.noise_scale *= self.noise_scale_decay
                self.lr_scale *= self.lr_decay
                opt.step()
                opt.zero_grad()
                    # print(torch.nn.NLLLoss()(self.model.forward(data),target))
                # for param in self.model.parameters():
                #
                #     print(param.data.abs().mean())
                # get_memory()

            print("Total Accuracy:", total_accuracy / batch_count,
                  "Average Reward:", total_reward / batch_count)
            fname = os.path.join(self.checkpoints_dir, "epoch_"+str(epoch)+".pkl")
            # perturbed_model.free_memory()
            # torch.save(self.model, fname)





if __name__ == "__main__":
    batch_size = 2 ** 14
    directions = batch_size
    # t = torch.zeros( int(1.5 * 2 ** 30), device="cuda")
    my_model = models.MNISTConvNet(directions=directions, action_size=10,in_channels=1).to(device)
    # my_model = models.MNISTDenseNet(directions=directions, action_size=10,in_channels=1)
    # my_model = models.VGG11(directions=directions, action_size=10, in_channels=24)
    # my_model = models.SmallNet(directions=directions, action_size=10, in_channels=24)
    # my_model = models.ResNet9(directions=directions, action_size=10, in_channels=3).to(device)
    print(my_model)
    # Trainer(model.TransformerNet()).train()
    Trainer(my_model, batch_size=batch_size, directions=directions).train()



