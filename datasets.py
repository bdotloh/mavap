import itertools
import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_files(split=1):
    files = sorted([file for file in os.listdir('data/50salads/groundtruth') if file.endswith('.txt')])

    kfold = KFold(5, shuffle=True, random_state=42)
    train_splits = []
    test_splits = []

    for train, test in kfold.split(files):
        train_splits.append([files[i] for i in train])
        test_splits.append([files[i] for i in test])

    train_files_for_curr_split = train_splits[int(split)-1]
    test_files_for_curr_split = test_splits[int(split)-1]

    return train_files_for_curr_split, test_files_for_curr_split


class FiftySaladsTraining(Dataset):
    def __init__(self, files):
        super().__init__()
        # store act to index mapping in dictionary
        self.act2ix = {}
        with open('data/50salads/mapping_50salads.txt', 'r') as file_ptr:
            ixs_acts = file_ptr.read().split('\n')[:-1]

        for ix_act in ixs_acts:
            ix_act = ix_act.split()
            ix = int(ix_act[0])
            act = ix_act[1]
            self.act2ix[act] = ix

        self.ix2act = dict([(value, key) for key, value in self.act2ix.items()])

        self.n_acts = len(self.act2ix)

        self.all_acts = []
        self.all_durs = []
        for i, file in enumerate(files):
            with open('data/50salads/groundtruth/{}'.format(file), 'r') as file_ptr:
                act_seq = file_ptr.read().split('\n')[:-1]
                acts = []
                durs = []
                for act, g in itertools.groupby(act_seq):
                    acts.append(self.act2ix[act])
                    dur = len(list(g))
                    durs.append(dur)

                for j in range(2, len(acts) + 1):
                    self.all_acts.append(acts[:j])
                    self.all_durs.append(durs[:j])

            durs_flatten = [dur for seq in self.all_durs for dur in seq]

            self.dur_mean = np.mean(durs_flatten)
            self.dur_std = np.std(durs_flatten)

    def __len__(self):
        return len(self.all_acts)

    def __getitem__(self, ix):
        acts = torch.tensor(self.all_acts[ix], dtype=torch.long, device=device)
        durs = torch.tensor([(dur - self.dur_mean) / self.dur_std for dur in self.all_durs[ix]],
                            dtype=torch.float, device=device)

        return acts, durs


class FiftySaladsTest(Dataset):
    def __init__(self, files):
        super().__init__()
        # store act to index mapping in dictionary
        self.act2ix = {}
        with open('data/50salads/mapping_50salads.txt', 'r') as file_ptr:
            ixs_acts = file_ptr.read().split('\n')[:-1]

        for ix_act in ixs_acts:
            ix_act = ix_act.split()
            ix = int(ix_act[0])
            act = ix_act[1]
            self.act2ix[act] = ix

        self.ix2act = dict([(value, key) for key, value in self.act2ix.items()])

        self.n_acts = len(self.act2ix)
        self.all_acts = []
        self.all_durs = []
        self.all_files = []
        for i, file in enumerate(files):
            self.all_files.append(file.split('.')[0])
            with open('data/50salads/groundtruth/{}'.format(file), 'r') as file_ptr:
                act_seq = file_ptr.read().split('\n')[:-1]
                acts = []
                durs = []
                for act, g in itertools.groupby(act_seq):
                    acts.append(self.act2ix[act])
                    dur = len(list(g))
                    durs.append(dur)
                self.all_acts.append(acts)
                self.all_durs.append(durs)

        durs_flatten = [dur for seq in self.all_durs for dur in seq]
        self.dur_mean = np.mean(durs_flatten)
        self.dur_std = np.std(durs_flatten)

    def __len__(self):
        return len(self.all_acts)

    def __getitem__(self, ix):
        acts = torch.tensor(self.all_acts[ix], dtype=torch.long, device=device)
        durs = torch.tensor(self.all_durs[ix], dtype=torch.float, device=device)
        return acts, durs


class BreakfastTraining(Dataset):
    def __init__(self, files):
        super().__init__()
        # store act to index mapping in dictionary
        self.act2ix = {}
        with open('data/breakfasts/mapping_breakfasts.txt', 'r') as file_ptr:
            ixs_acts = file_ptr.read().split('\n')[:-1]

        for ix_act in ixs_acts:
            ix_act = ix_act.split()
            ix = int(ix_act[0])
            act = ix_act[1]
            self.act2ix[act] = ix

        self.ix2act = dict([(value, key) for key, value in self.act2ix.items()])

        self.all_acts = []
        self.all_durs = []
        self.all_files = []
        for i, file in enumerate(files):
            self.all_files.append(file)
            with open('data/breakfasts/groundtruth/{}'.format(file), 'r') as file_ptr:
                act_seq = file_ptr.read().split('\n')[:-1]
                acts = []
                durs = []
                for act, g in itertools.groupby(act_seq):
                    acts.append(self.act2ix[act])
                    dur = len(list(g))
                    durs.append(dur)

                for j in range(2,len(acts)+1):
                    self.all_acts.append(acts[:j])
                    self.all_durs.append(durs[:j])

        durs_flatten = [dur for seq in self.all_durs for dur in seq]

        self.dur_mean = np.mean(durs_flatten)
        self.dur_std = np.std(durs_flatten)

    def __len__(self):
        return len(self.all_acts)

    def __getitem__(self, ix):
        acts = torch.tensor(self.all_acts[ix], dtype=torch.long, device=device)
        durs = torch.tensor([(dur - self.dur_mean) / self.dur_std for dur in self.all_durs[ix]],
                            dtype=torch.float, device=device)

        return acts, durs


class BreakfastTest(Dataset):
    def __init__(self, files):
        super().__init__()
        # store act to index mapping in dictionary
        self.act2ix = {}
        with open('data/breakfasts/mapping_breakfasts.txt', 'r') as file_ptr:
            ixs_acts = file_ptr.read().split('\n')[:-1]

        for ix_act in ixs_acts:
            ix_act = ix_act.split()
            ix = int(ix_act[0])
            act = ix_act[1]
            self.act2ix[act] = ix

        self.ix2act = dict([(value, key) for key, value in self.act2ix.items()])

        self.all_acts = []
        self.all_durs = []
        self.all_files = []
        for i, file in enumerate(files):
            self.all_files.append(file)
            with open('data/breakfasts/groundtruth/{}'.format(file), 'r') as file_ptr:
                act_seq = file_ptr.read().split('\n')[:-1]
                acts = []
                durs = []
                for act, g in itertools.groupby(act_seq):
                    acts.append(self.act2ix[act])
                    dur = len(list(g))
                    durs.append(dur)

                self.all_acts.append(acts)
                self.all_durs.append(durs)

        durs_flatten = [dur for seq in self.all_durs for dur in seq]
        self.dur_mean = np.mean(durs_flatten)
        self.dur_std = np.std(durs_flatten)

    def __len__(self):
        return len(self.all_acts)

    def __getitem__(self, ix):
        acts = torch.tensor(self.all_acts[ix], dtype=torch.long, device=device)
        durs = torch.tensor(self.all_durs[ix], dtype=torch.float, device=device)
        return acts, durs


def split_sequence(acts, durs, obs=.2, pred=.5):
    total_obs_dur = torch.round(torch.sum(durs) * obs).item()
    dur_so_far = 0
    durs_list = durs[0].detach().tolist()

    for ix in range(len(durs_list)):
        dur_so_far += durs_list[ix]
        if dur_so_far > total_obs_dur:
            obs_durs = durs_list[:ix + 1]
            fut_durs = durs_list[ix:]
            diff = dur_so_far - total_obs_dur
            obs_durs[-1] -= diff
            fut_durs[0] -= obs_durs[-1]

            obs_acts = acts[:, :ix + 1]
            fut_acts = acts[:, ix:]
            break
    if pred == 1:
        return (obs_acts, obs_durs), (fut_acts, fut_durs)
    else:
        return (obs_acts, obs_durs), cut_future_by_pred_percent(fut_acts, fut_durs, pred)


def split_list_sequence(acts, durs, obs=.2):
    # total_obs_dur = torch.round(torch.sum(durs) * obs).item()
    total_obs_dur = round(sum(durs) * obs)
    dur_so_far = 0
    # durs_list = durs[0].detach().tolist()
    durs_list = durs

    for ix in range(len(durs_list)):
        dur_so_far += durs_list[ix]
        if dur_so_far > total_obs_dur:
            obs_durs = durs_list[:ix + 1]
            fut_durs = durs_list[ix:]
            diff = dur_so_far - total_obs_dur
            obs_durs[-1] -= diff
            fut_durs[0] -= obs_durs[-1]

            obs_acts = acts[:ix + 1]
            fut_acts = acts[ix:]
            break

    return obs_acts + fut_acts, obs_durs + fut_durs


def cut_future_by_pred_percent(acts, durs, pred=1):
    total_pred_dur = round(sum(durs) * pred)
    dur_so_far = 0

    for ix in range(len(durs)):
        dur_so_far += durs[ix]
        if dur_so_far > total_pred_dur:
            fut_durs = durs[:ix + 1]
            diff = dur_so_far - total_pred_dur
            fut_durs[-1] -= diff
            fut_acts = acts[:, :ix + 1]
            break
    return (fut_acts, fut_durs)


def pad_collate(batch):
    obs_act_seqs = []
    obs_dur_seqs = []

    tar_acts = []
    tar_durs = []
    act_seqs, dur_seqs = zip(*batch)
    for act_seq, dur_seq in zip(act_seqs, dur_seqs):
        obs_act_seqs.append(act_seq[:-1])
        obs_dur_seqs.append(dur_seq[:-1])

        tar_acts.append(act_seq[-1])
        tar_durs.append(dur_seq[-1])

    obs_acts_pad = pad_sequence(obs_act_seqs, batch_first=True, padding_value=0)
    obs_durs_pad = pad_sequence(obs_dur_seqs, batch_first=True, padding_value=0)

    return (obs_acts_pad, obs_durs_pad), (torch.tensor(tar_acts),torch.tensor(tar_durs))



trainset,testset = get_files()
dataset = FiftySaladsTraining(trainset)
print(len(dataset))
dataloader = DataLoader(dataset,batch_size=2, collate_fn=pad_collate)

print('-----train-----')
obs, tar = next(iter(dataloader))
print(obs)
print(tar)

testset = FiftySaladsTest(testset)
testloader = DataLoader(testset, batch_size=1)

# print('-----test-----')
# acts, durs = next(iter(testloader))
# print('sequence before split: \n')
# print(acts, durs)
# print('sequence after split: \n')
# obs_seq, fut_seq = split_sequence(acts, durs, .3, .5)
# print(f'obs seq {obs_seq} \n fut seq {fut_seq}')
#
# obs_acts, obs_dur = obs_seq
# fut_acts, fut_durs = fut_seq

# total_dur = sum(fut_durs)
# pred_acts, pred_durs = model.generate(obs_acts, obs_dur, total_dur,testset.dur_mean,testset.dur_std)
# print(pred_acts,pred_durs)
