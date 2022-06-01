from dataset_breakfast import *
from model import MAVAP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(dataset, batch_size, split, h_dim, z_dim, n_layers, n_heads, epochs, checkpoint_path=None):
    if dataset == 'breakfast':
        file_dir = 'data/breakfast'
        trainset = BreakfastseqDataset(file_dir, split, 'train')
        act_dim = 48

    model = MAVAP(act_dim=act_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers,
                  n_heads=n_heads)
    model.train()

    optimizer = Adam(model.parameters(), lr=.00001)
    cross_entropy = torch.nn.CrossEntropyLoss()
    gaussian_nll = torch.nn.GaussianNLLLoss()
    scheduler = CosineAnnealingLR(optimizer, len(trainset))

    dataloader = DataLoader(trainset, batch_size=batch_size, collate_fn=seq_collate_dict_train, shuffle=False)

    epoch_start = 1
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        epoch_start = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(epoch_start, epoch_start + epochs):
        print(f'----------epoch {epoch}-------')
        running_kld = 0.0
        running_act_loss = 0.0
        running_dur_loss = 0.0

        for i, data in enumerate(dataloader):
            batch, mask, length = data
            obs_acts = batch['obs_act_seqs']
            obs_durs = batch['obs_dur_seqs']
            tar_act = batch['pred_act']
            tar_dur = batch['pred_dur']

            pred_act, pred_dur, priors, posteriors = model(obs_acts, obs_durs)

            pred_dur_mean, pred_dur_std = pred_dur
            prior_means, prior_stds = priors
            posterior_means, posterior_stds = posteriors

            act_loss = cross_entropy(pred_act, tar_act)
            dur_loss = gaussian_nll(pred_dur_mean, tar_dur, pred_dur_std)
            kl_loss = 0
            T = len(posteriors[0])
            for t in range(T):
                kl_t = kld_gauss(
                    prior_means[t],
                    prior_stds[t],
                    posterior_means[t],
                    posterior_stds[t],
                    mask=mask[:, t, :]
                )
                kl_loss += kl_t

            running_kld += kl_loss.item()
            running_act_loss += act_loss.item()
            running_dur_loss += dur_loss.item()

            loss = act_loss + dur_loss + kl_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss_kl = running_kld / (i + 1)
        epoch_loss_act = running_act_loss / (i + 1)
        epoch_loss_dur = running_dur_loss / (i + 1)

        print(epoch_loss_dur + epoch_loss_act + epoch_loss_kl)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="breakfast",
                        help='breakfast or 50salad')
    parser.add_argument('--split', type=int, default=1,
                        help='data split: 1 to 5')
    parser.add_argument('-bs', '--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('-hid', '--h-dim', type=int, default=64,
                        help='hidden dimension')
    parser.add_argument('-z', '--z-dim', type=int, default=16,
                        help='z dimension')
    parser.add_argument('-l', '--n-layers', type=int, default=1,
                        help='number of dimension')
    parser.add_argument('-head', '--n-heads', type=int, default=4,
                        help='number of heads')
    parser.add_argument('--epochs', type=int, default=2)

    args = parser.parse_args()

    train(dataset=args.dataset,
          split=args.split,
          batch_size=args.batch_size,
          h_dim=args.h_dim,
          z_dim=args.z_dim,
          n_layers=args.n_layers,
          n_heads=args.n_heads,
          epochs=args.epochs)