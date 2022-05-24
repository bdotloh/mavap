from utils import *
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

from model import VRNN


class BreakfastseqDataset(Dataset):
    def __init__(self, root_dir, split, split_type):  # change to args
        super().__init__()
        self.act_dict = read_mapping_dict(os.path.join(root_dir, 'mapping_breakfast.txt'))
        self.data = {'act_seqs_one_hot': [], 'act_seqs_ix': [], 'dur_seqs': [], 'seq_lens': []}

        with open(os.path.join(root_dir, f'{split_type}.split{split}.bundle'), 'r') as file_ptr_0:
            video_list = file_ptr_0.read().split('\n')[:-1]
            for video_name in video_list:
                # goal = video_name.split('.')[0].split('_')[-1]
                with open(os.path.join(root_dir, f'groundtruth/{video_name}'), 'r') as file_ptr_1:
                    content = file_ptr_1.read().split('\n')[:-1]
                    action_seq, duration_seq = get_label_length_seq(content)
                    action_ix_seq = torch.tensor([self.act_dict[ix] for ix in action_seq], dtype=torch.long)
                    action_ix_seq_one_hot = one_hot(action_ix_seq, num_classes=len(self.act_dict))
                    self.data['act_seqs_ix'].append(action_ix_seq.unsqueeze(-1))
                    self.data['act_seqs_one_hot'].append(action_ix_seq_one_hot)
                    self.data['dur_seqs'].append(torch.tensor(duration_seq).unsqueeze(-1))
                    self.data['seq_lens'].append(len(action_seq))

    def __len__(self):
        return len(self.data['act_seqs_one_hot'])

    def __getitem__(self, i):
        return {k: self.data[k][i] for k in ['act_seqs_one_hot', 'act_seqs_ix', 'dur_seqs', 'seq_lens']}

    def decode_act_ix_sequence(self, action_ix_sequence):
        return [list(self.act_dict.keys())[list(self.act_dict.values()).index(act_ix)] for act_ix in action_ix_sequence]


def seq_collate_dict(data, time_first=False):
    """Collate that accepts and returns dictionaries."""
    batch = {}
    modalities = [k for k in data[0].keys() if k != 'seq_lens']
    data.sort(key=lambda d: d['seq_lens'], reverse=True)
    lengths = [d['seq_lens'] for d in data]
    for m in modalities:
        m_data = [d[m] for d in data]
        m_padded = pad_and_merge(m_data, max(lengths))
        batch[m] = m_padded.permute(1, 0, 2) if time_first else m_padded
    mask = len_to_mask(lengths).unsqueeze(-1)
    if time_first:
        mask = mask.permute(1, 0, 2)
    return batch, mask, lengths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="data/breakfast",
                        help='data directory')
    parser.add_argument('--split', type=int, default=1,
                        help='data split: 1 to 5')
    parser.add_argument('--split-type', type=str, default="train",
                        help='train or test')
    parser.add_argument('-bs', '--batch-size', type=int, default=2,
                        help='batch size')
    parser.add_argument('-act', '--act-dim', type=int, default=48,
                        help='act dimension')
    parser.add_argument('-hid', '--h-dim', type=int, default=64,
                        help='hidden dimension')
    parser.add_argument('-z', '--z-dim', type=int, default=16,
                        help='z dimension')
    parser.add_argument('-l', '--n-layers', type=int, default=1,
                        help='number of dimension')
    parser.add_argument('-head', '--n-heads', type=int, default=4,
                        help='number of heads')

    args = parser.parse_args()
    dataset = BreakfastseqDataset(args.dir, args.split, args.split_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=seq_collate_dict)
    example_batch, mask, length = next(iter(dataloader))
    print(example_batch['act_seqs_one_hot'].shape)
    model = VRNN(act_dim=args.act_dim, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=args.n_layers)
    priors, posteriors, pred_acts, pred_durs = model(example_batch['act_seqs_one_hot'], example_batch['dur_seqs'])
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    nll_gauss = torch.nn.GaussianNLLLoss(reduction='none')
    ce_loss = 0
    kl_loss = 0
    nll_gauss_loss = 0
    T = example_batch['act_seqs_ix'].size(1)
    for t in range(T):
        print(kl_loss)
        kl_loss += kld_gauss(
            posteriors[0][:, t, :],
            posteriors[-1][:, t, :],
            priors[0][:, t, :],
            priors[-1][:, t, :],
            mask=mask[:, t, :]
        )
        ce_loss += cross_entropy(
            pred_acts[:, t, :],
            example_batch['act_seqs_ix'][:, t, :].type(torch.long).squeeze(-1)).masked_select(
            mask[:, t, :].squeeze(-1)).sum()
        nll_gauss_loss += nll_gauss(
            pred_durs[0][:, t, :],
            example_batch['dur_seqs'][:, t, :],
            pred_durs[-1][:, t, :]).masked_select(
            mask[:, t, :].squeeze(-1)).sum()

    ce_loss /= args.batch_size
    kl_loss /= args.batch_size
    nll_gauss_loss /= args.batch_size
    print(ce_loss, kl_loss, nll_gauss_loss)
