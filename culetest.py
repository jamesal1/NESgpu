from torchcule.atari import Env
import models
import modules


from tqdm import trange
import os
import datetime
import shutil
import torch
import random
random.seed(2)
torch.manual_seed(2)
import time
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
        for f in ["culetest.py", "models.py"]:
            src = os.path.join(os.getcwd(), f)
            dst = os.path.join(self.log_dir, f)
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
        self.batches_per_epoch = kwargs.get("batches_per_epoch",10)
        self.batch_size = kwargs.get("batch_size",1)
        self.directions = kwargs.get("directions",1)
        self.max_steps = kwargs.get("max_steps", -1)

    def train(self):
        self.model.batch_size=self.batch_size
        if half_precision:
            self.model = self.model.half()
        perturbed_model = modules.PerturbedModel(self.model,self.directions)
        ave_delta = .005 * self.batch_size
        # opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay, eps=1e-3)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)



        for epoch in range(self.epochs):
            print("Epoch:",epoch)
            total_reward = 0.0
            total_game_length = 0.0
            total_cards = 0.0
            total_points = 0.0
            max_cards = 0
            max_score = 0
            for _ in trange(self.batches_per_epoch):
                with torch.no_grad():
                    repeat_size = self.batch_size // self.directions
                    perturbed_model.set_seed()
                    perturbed_model.set_noise_scale(self.noise_scale)
                    perturbed_model.allocate_memory(low_memory=False)
                    # perturbed_model.allocate_memory(low_memory=False)
                    perturbed_model.set_noise()
                    observations = env.reset()
                    step = 0
                    sum_reward = torch.zeros(self.batch_size, device=cuda_device)
                    while step != self.max_steps:
                        start = time.time()
                        modules.print_all_torch_tensors()
                        actions = perturbed_model.forward(observations.permute([0,3,1,2]).type(precision))
                        modules.print_all_torch_tensors()
                        time.sleep(1000)
                        print("nn time", time.time() - start)
                        start = time.time()
                        observations, reward, done, info = env.step(actions)
                        print("step time", time.time() - start)
                        sum_reward.add_(reward.float())
                        step += 1
                        if step % 100 ==0:
                            print(step)
                        if torch.all(done):
                            break





                    result = result.view(repeat_size, self.directions)
                    reward_delta = result[repeat_size // 2:].sum(dim=0) - result[:repeat_size // 2].sum(dim=0)
                    step_size = reward_delta / ((ave_delta + 1e-5) * self.noise_scale)
                    ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * (reward_delta.norm(p=1))
                    total_reward += sum_reward.mean()

                perturbed_model.set_grad(step_size)
                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())
                self.noise_scale *= self.noise_scale_decay
                opt.step()
            # for param in self.model.parameters():
            #
            #     print(param.data.abs().mean())
            print("Average Reward:", total_reward / (self.batches_per_epoch))
            # print("Average Game Length:", total_game_length.float() / (self.batches_per_epoch * self.batch_size))
            fname = os.path.join(self.checkpoints_dir, "epoch_"+str(epoch)+".pkl")
            perturbed_model.free_memory()
            torch.save(self.model, fname)





if __name__ == "__main__":
    batch_size = 2 ** 11
    directions = 2 ** 10

    game = "PongNoFrameskip-v4"
    color_mode = "gray"
    colors = 1 if color_mode == "gray" else 3
    env = Env(game, batch_size, color_mode, torch.device("cuda",0), True)
    print(env.action_space)
    my_model = models.ConvNet(directions=directions, action_size=6,in_channels=colors)
    # Trainer(model.TransformerNet()).train()
    if cuda_on:
        my_model = my_model.cuda()
    Trainer(my_model,batch_size=batch_size,directions=directions).train()



