import torch

from dataset_breakfast import *
from model import MAVAP


def eval(dataset, split, obs_per, pred_per, h_dim, z_dim, n_layers, n_heads, checkpoint_path=None):
    if dataset == 'breakfast':
        file_dir = 'data/breakfast'
        testset = BreakfastDataset(file_dir, split, 'test')
        act_dim = 48

    model = MAVAP(act_dim=act_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers,
                  n_heads=n_heads)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    dataloader = DataLoader(testset, batch_size=1, collate_fn=seq_collate_dict, shuffle=False)

    all_tar = []
    all_pred = []
    for data in dataloader:
        act_seq, dur_seq = data[0]['act_seqs_ix'], data[0]['dur_seqs']
        (obs_acts, obs_durs), (tar_future_acts, tar_future_durs) = split_sequence(
            act_seq,
            dur_seq,
            obs=obs_per,
            pred=pred_per)

        # process model inputs
        obs_acts_one_hot = one_hot(obs_acts, num_classes=act_dim)
        obs_durs_norm = obs_durs.apply_(lambda x: (x - testset.dur_mean)/testset.dur_std).unsqueeze(-1) # normalise duration
        total_dur_to_predict = sum(tar_future_durs)

        # predict
        pred_acts, pred_durs = model.forecast(obs_acts_one_hot, obs_durs_norm, total_dur_to_predict, testset.dur_mean, testset.dur_std)
        pred_framewise = list(np.repeat(pred_acts, pred_durs, axis=0))
        tar_framewise = list(np.repeat(tar_future_acts.view(-1).detach().tolist(), tar_future_durs, axis=0))

        # post-process: ensure both pred and groundtruth framewise lengths are similar
        if len(pred_framewise) > len(tar_framewise):
            pred_framewise = pred_framewise[:len(tar_framewise)]

        elif len(pred_framewise) < len(tar_framewise):
            diff = len(tar_framewise) - len(pred_framewise)
            pred_framewise.extend([pred_framewise[-1]] * diff)

        all_tar.extend(tar_framewise)
        all_pred.extend(pred_framewise)

    print(balanced_accuracy_score(all_tar, all_pred))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="breakfast",
                        help='breakfast or 50salad')
    parser.add_argument('--split', type=int, default=1,
                        help='data split: 1 to 5')
    parser.add_argument('-hid', '--h-dim', type=int, default=64,
                        help='hidden dimension')
    parser.add_argument('-z', '--z-dim', type=int, default=16,
                        help='z dimension')
    parser.add_argument('-l', '--n-layers', type=int, default=1,
                        help='number of rnn layers')
    parser.add_argument('-head', '--n-heads', type=int, default=4,
                        help='number of self-attention heads')
    parser.add_argument('-obs', '--obs-per', type=float, default=.2,
                        help='percentage of video that model observes')
    parser.add_argument('-pred', '--pred-per', type=float, default=.5,
                        help='percentage of video that model predicts')

    args = parser.parse_args()

    eval(dataset=args.dataset,
         split=args.split,
         obs_per=args.obs_per,
         pred_per=args.pred_per,
         h_dim=args.h_dim,
         z_dim=args.z_dim,
         n_layers=args.n_layers,
         n_heads=args.n_heads,
         checkpoint_path=None)