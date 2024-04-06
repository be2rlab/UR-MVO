import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SP.dataset import AQUADataset
from SP.loss import descriptor_loss
from SP.model import SuperPoint


def train(model, dataloader, config):
    optimized_params = list(model.convDa.parameters()) + list(model.convDb.parameters())
    # optimized_params = model.parameters()
    optimizer = torch.optim.Adam(optimized_params, lr=config['base_lr'])
    try:
        for epoch in range(config['epoch']):
            model.train()
            epoch_loss = []
            inter_loss = []
            for i, data in tqdm(enumerate(dataloader)):
                optimizer.zero_grad()
                image_orig = data['raw']['img'].cuda()
                _, descriptors_dense_orig = model(image_orig)
                image_warp = data['warp']['img'].cuda()
                _, descriptors_dense_warp = model(image_warp)
                homography = data['homography']
                w_mask = data['warp']['mask']
                loss = descriptor_loss(descriptors_dense_orig.cuda(),
                                       descriptors_dense_warp.cuda(),
                                       homography.cuda(),
                                       valid_mask=w_mask.cuda(),
                                       device='cuda')
                epoch_loss.append(loss.item())
                inter_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch}/{config['epoch']}]  Loss: {np.mean(epoch_loss):.3f}")
            epoch_loss = []

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "./weights/model_interrupt.pth")
    else:
        torch.save(model.state_dict(), "./weights/model_finished.pth")


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset=AQUADataset(enable_homo=True,
                                                enable_photo=True,
                                                device='cpu'
                                                ),
                            batch_size=2,
                            shuffle=True,
                            )
    print(dataloader)

    config = {'base_lr': 0.001,
              'epoch': 100,
              'log_interval': 10}

    model = SuperPoint().to(device)
    model.load_state_dict(torch.load('./weights/superpoint_v1.pth'))
    train(model, dataloader, config)
