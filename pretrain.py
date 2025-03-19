from tqdm import tqdm
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from RL.utils import device

from game.fb_game import FlappyBird
from game.utils import window_height, window_width, fps
from game.utils import window_height, window_width, fps

class PretrainModel(nn.Module):
    def __init__(self, model, learn_features=False):
        super().__init__()
        self.model = model
        self.learn_features = learn_features
        if not learn_features: self.fc = nn.Linear(model.repr_dim, 6)     
    
    def forward(self, state):
        representation = self.model.pretrain_forward(state)
        if not self.learn_features : representation = self.fc(representation)
        return representation
    
    def freeze(self):
        self.model.freeze_pretrain()

    def train_mode(self):
        self.train()
        self.model.train()
    
    def eval_mode(self):
        self.eval()
        self.model.eval()


def pretrain_features(game):
    next_top_pipes = [pipe for pipe in game.pipes.top_pipes if game.bird.pos.x < (pipe.pos.x + pipe.size[0])]
    next_bottom_pipes = [pipe for pipe in game.pipes.bottom_pipes if game.bird.pos.x < (pipe.pos.x + pipe.size[0])]
    ### get features
    # top pipe
    top_pipe_x = next_top_pipes[0].pos.x if len(next_top_pipes)>0 else window_width - game.bird.pos.x
    top_pipe_y = next_top_pipes[0].pos.y + 611 if len(next_top_pipes)>0 else window_height/2
    # bottom pipe
    bottom_pipe_x = next_bottom_pipes[0].pos.x if len(next_bottom_pipes)>0 else window_width - game.bird.pos.x
    bottom_pipe_y = next_bottom_pipes[0].pos.y if len(next_bottom_pipes)>0 else window_height/2
    # bird
    bird_y = game.bird.pos.y
    bird_angle = game.bird.angle
    ### manual normalization
    # top pipe
    top_pipe_x = (top_pipe_x - (window_width - game.bird.pos.x)/2) / window_width
    top_pipe_y = (top_pipe_y - window_height/2) / window_height
    # bottom pipe
    bottom_pipe_x = (bottom_pipe_x - (window_width - game.bird.pos.x)/2) / window_width
    bottom_pipe_y = (bottom_pipe_y - window_height/2) / window_height
    bird_y = (bird_y - window_height/2) / window_height
    bird_angle = (bird_angle) / 90

    return [bird_y, bird_angle, top_pipe_x, top_pipe_y, bottom_pipe_x, bottom_pipe_y]


def generate_dataset(agent, dataset_size, nframes):  
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=True)
    Xs, Ys = [], []
    for _ in tqdm(range(dataset_size)):
        state = game.set_random_state()
        for _ in range(nframes-1):
            state, _, _, _ = game.step(1 if np.random.random()<0.1 else 0)
        Xs.append(state)
        Ys.append(pretrain_features(game))
    return torch.tensor(Xs, dtype=torch.float), torch.tensor(Ys, dtype=torch.float)


def train(pretrain_model, Xs, Ys, epochs, batch_size, **optim_kwargs):
    print("Pretraining Start")
    optimizer = optim.Adam(pretrain_model.parameters(), **optim_kwargs)
    criterion = nn.MSELoss() # lambda pred, target: (((pred - target)/(target + 1e-3))**2).mean()  
    dataset = torch.utils.data.TensorDataset(Xs, Ys)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pretrain_model.train_mode()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = pretrain_model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Pretrain Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    print("Pretraining Done")


def eval(pretrain_model, Xs_val, Ys_val):
    criterion = nn.MSELoss()
    pretrain_model.eval_mode()
    with torch.no_grad():
        Xs_val, Ys_val = Xs_val.to(device), Ys_val.to(device)
        Ys_pred = pretrain_model.forward(Xs_val)
        val_loss = criterion(Ys_pred, Ys_val).item()
    pretrain_model.train_mode()
    print(f"Validation loss, {val_loss}")


def pretrain(agent, epochs=10, dataset_size=1000, batch_size=64, 
             save_dataset=True, use_saved=True, dataset_path="pretrained_dataset.pth", 
             nframes=1, learn_features=False, freeze=True, **optim_kwargs):

    if not use_saved or not os.path.exists(dataset_path): 
        print("Generating train dataset")
        Xs, Ys = generate_dataset(agent, dataset_size, nframes)
        if save_dataset: torch.save({'Xs': Xs, 'Ys': Ys}, dataset_path)
    else:
        dataset = torch.load(dataset_path)
        Xs = dataset["Xs"]
        Ys = dataset["Ys"]

    pretrain_model = PretrainModel(agent.policy, learn_features).to(device)
    train(pretrain_model, Xs , Ys, epochs , batch_size, **optim_kwargs)

    print("Generating validation dataset")
    Xs_val, Ys_val = generate_dataset(agent, 250, nframes)
    eval(pretrain_model, Xs_val, Ys_val)

    if freeze: pretrain_model.freeze()
    print("Froze pretrained model")

    return pretrain_model