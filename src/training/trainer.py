from torch.nn.utils import clip_grad_norm_
from training.training_config import Config as train_cfg

device = train_cfg.DEVICE


def train_one_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    optimizer.zero_grad()

    for i, (inputs, subject, targets, key, mask) in enumerate(train_dataloader):
        inputs = inputs.to(device)            # [B, C, T]
        targets = targets.to(device)           # [B, C, T]
        mask = mask.to(device)              # [B, T]
        # forward
        outputs = model(inputs, subject)       # [B, C, T]
        outputs = outputs.permute(0, 2, 1)     # [B, T, C]
        targets = targets.permute(0, 2, 1)     # [B, T, C]
        # 2) flatten B & T into one dim
        B, T, C = outputs.shape
        outputs_flat = outputs.reshape(-1, C)  # [B*T, C]
        targets_flat = targets.reshape(-1, C)  # [B*T, C]

        # 3) flatten mask to pick only valid timesteps
        mask_flat = mask.view(-1).bool()       # [B*T]

        # 4) select valid rows
        outputs_sel = outputs_flat[mask_flat]  # [#valid, C]
        targets_sel = targets_flat[mask_flat]  # [#valid, C]
        # 5) compute loss just on valid timesteps
        loss = criterion(outputs_sel, targets_sel).mean()  

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        optimizer.zero_grad()

    return model
