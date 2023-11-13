import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

import model
import data_setup
import utils

def train_step(model: torch.nn.Module,
               dataset: torch.utils.data.Dataset,
               batch_sz: int,
               context_len: int,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    model.train()
    steps = 1
    losses_g = []
    for st in range(steps):
        ixs = torch.randint(3, len(dataset), (batch_sz,), device=device)
        data = dataset[ixs]

        targets = torch.randint(5, 50, (batch_sz,), device=device).unsqueeze(0).repeat(context_len, 1)  # Scales up the dimensionality of the targets so they can be sampled in one line
        context_locs = torch.normal(mean=targets.float(), std=3).long().clamp_(0, 54)  # Samples from a normal distribution (means must be float). Finally clamped so no negative values or that funny buisness

        times = ixs.unsqueeze(0).repeat(4, 1)
        context_times = torch.normal(mean=times.float(), std=1.5).long().clamp_(0, len(dataset)-1).to(device)

        context_ix = torch.permute(torch.cat((context_locs.unsqueeze(0), context_times.unsqueeze(0)), dim=0), (2, 1, 0))
        targets_ix = torch.cat((targets[0].unsqueeze(0), times[0].unsqueeze(0)), dim=0).transpose(1, 0)
        
        context_ix = utils.validate_context_and_fix(context_ix, targets_ix, batch_sz)
        context = torch.stack([dataset[rq.transpose(1, 0)[1] % (len(dataset) - 1), rq.transpose(1, 0)[0]] for rq in context_ix])

        normalized_time = (context_ix[:, :, 1] - times.transpose(1, 0)).long() # Make times fit into the embedding layer lmao
        target_positions = targets_ix[:, 0]

        out = model(context, context_ix[:, :, 0].to(device), normalized_time.to(device), target_positions.to(device)).double()

        sol = torch.stack([scan[target_positions[i]] for i, scan in enumerate(data)])

        loss = loss_fn(out, sol.view(-1, 64*64))
        optimizer.zero_grad()
        losses_g.append(loss.item())
        if st % 5 == 0:
          print(f"Step [{st+1}], Loss: {sum(losses_g)/len(losses_g)}")
          losses_g = []


if __name__ == "__main__":
    conf = model.Configs()
    context_len = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.Model(conf).to(device)
    dataset = data_setup.create_dataset(1, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_step(model, dataset, batch_sz=5, context_len=4, loss_fn=criterion, optimizer=optimizer, device=device)


