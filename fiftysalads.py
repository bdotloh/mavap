from pathlib import Path
import pandas as pd
from model import *
from datasets import *

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import KFold
from sklearn.metrics import top_k_accuracy_score, balanced_accuracy_score

torch.manual_seed(42)

epochs = 20
n_z = 12
h_dim = 64
z_dim = 64

files = sorted([file for file in os.listdir('data/50salads/groundtruth') if file.endswith('.txt')])
kfold = KFold(5, shuffle=True, random_state=42)


def fiftysalads_train(h_dim, z_dim, n_z, epochs, checkpoint_path=None):
    Path("50-salads-diagnostics/n_z_{}/training".format(n_z)).mkdir(parents=True,exist_ok=True)
    Path("50-salads-diagnostics/n_z_{}/saved_models".format(n_z)).mkdir(parents=True, exist_ok=True)
    for fold, (train, test) in enumerate(kfold.split(files)):
        trainset = FiftySaladsTraining([files[i] for i in train])

        print(f'Fold: {fold + 1} \n train instances: {len(trainset)}')
        trainloader = DataLoader(trainset, batch_size=1, collate_fn=pad_collate, shuffle=True)

        model = AnticipationModel(act_dim=19, h_dim=h_dim, z_dim=z_dim, n_heads=n_z)
        optimizer = Adam(model.parameters(), lr=.0001)

        epoch_start = 1
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            epoch_start = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])

        scheduler = CosineAnnealingLR(optimizer, len(trainset))
        cel = nn.CrossEntropyLoss()
        gaussian_nll = nn.GaussianNLLLoss(full=True)

        model.train()

        total_losses = []
        kl_losses = []
        act_losses = []
        dur_losses = []
        top_1_accuracies = []
        top_5_accuracies = []

        highest_acc = 0
        for epoch in range(epoch_start, epoch_start+epochs):
            print(f'----------epoch {epoch}-------')
            running_kld = 0.0
            running_act_loss = 0.0
            running_dur_loss = 0.0

            tar_acts = np.zeros((len(trainset),))
            pred_acts_prob = np.zeros((len(trainset), 19))
            for i, data in enumerate(trainloader):
                obs, tar = data
                pred_act_prob_unnorm, pred_dur_params, kl_loss = model(*obs)

                tar_act = tar[0]
                tar_dur = tar[-1]

                tar_acts[i] = tar_act
                pred_acts_prob[i, :] = pred_act_prob_unnorm.detach().numpy()

                act_loss = cel(pred_act_prob_unnorm, tar_act)
                dur_loss = gaussian_nll(pred_dur_params[0], tar_dur, pred_dur_params[-1])

                loss = act_loss + kl_loss + dur_loss

                running_kld += kl_loss.item()
                running_act_loss += act_loss.item()
                running_dur_loss += dur_loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_loss_kl = running_kld / (i + 1)
            epoch_loss_act = running_act_loss / (i + 1)
            epoch_loss_dur = running_dur_loss / (i + 1)

            top_1_acc = top_k_accuracy_score(tar_acts, pred_acts_prob, k=1, labels=[i for i in range(19)])
            top_5_acc = top_k_accuracy_score(tar_acts, pred_acts_prob, k=5, labels=[i for i in range(19)])

            total_losses.append(epoch_loss_kl + epoch_loss_act + epoch_loss_dur)
            kl_losses.append(epoch_loss_kl)
            act_losses.append(epoch_loss_act)
            dur_losses.append(epoch_loss_dur)
            top_1_accuracies.append(top_1_acc)
            top_5_accuracies.append(top_5_acc)

            print('Epoch Completed!\n%d kl loss = %.3f, act loss = %.3f, dur loss= %.3f, top 1 acc= %.3f, top 5 acc= %.3f' %
                  (epoch,
                   epoch_loss_kl,
                   epoch_loss_act,
                   epoch_loss_dur,
                   top_1_acc,
                   top_5_acc
                   ))

            if top_5_acc > highest_acc:
                print('saving model...')
                path = '50-salads-diagnostics/n_z_{}/saved_models/50salads_fold{}_nz{}.pth'.format(str(n_z),
                                                                                               str(fold + 1),
                                                                                               str(n_z))
                highest_acc = top_5_acc
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'acc': top_5_acc
                            }, path)

        training_diagnostics = pd.DataFrame({'epoch': [i for i in range(1, epochs + 1)],
                                             'total_loss': total_losses,
                                             'kl_loss': kl_losses,
                                             'act_loss': act_losses,
                                             'dur_loss': dur_losses,
                                             'top_1': top_1_accuracies,
                                             'top_5': top_5_accuracies})

        training_diagnostics.to_csv('50-salads-diagnostics/n_z_{}/training/training_diagnostics_fold_{}.csv'.format(str(n_z), str(fold + 1)))

def fiftysalad_eval(h_dim, z_dim, n_z, checkpoint_path=None):
    all_accs = []

    acc_dict = {.2: {k: [] for k in [.1, .2, .3, .5]}, .3: {k: [] for k in [.1, .2, .3, .5]}}
    for fold, (train, test) in enumerate(kfold.split(files)):
        valset = FiftySaladsTest([files[i] for i in test])
        print(f'Fold: {fold + 1} val instances: {len(valset)}')

        valloader = DataLoader(valset, batch_size=1, shuffle=False)

        model = AnticipationModel(act_dim=19, h_dim=h_dim, z_dim=z_dim, n_heads=n_z)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        pred_dict = {k: [] for k in [.1, .2, .3, .5]}
        tar_dict = {k: [] for k in [.1, .2, .3, .5]}
        for obs_per in [.2, .3]:
            Path("50-salads-diagnostics/n_z_{}/results/obs{}".format(n_z, str(obs_per))).mkdir(parents=True,
                                                                                               exist_ok=True)
            for i, data in enumerate(valloader):
                file_name = valset.all_files[i]
                acts, durs = data
                obs_seq, fut_seq = split_sequence(acts, durs, obs_per, 1)
                obs_acts, obs_dur = obs_seq
                fut_acts, fut_durs = fut_seq
                obs_dur_norm = [(dur - valset.dur_mean) / valset.dur_std for dur in obs_dur]
                total_dur = sum(fut_durs)
                pred_acts, pred_durs = model.generate(obs_acts, obs_dur_norm, total_dur, valset.dur_mean,
                                                      valset.dur_std)

                # pred_durs = [dur if dur >= 0 else 1 for dur in pred_durs]

                obs_framewise = list(np.repeat(obs_acts.detach().numpy()[0], obs_dur, axis=0))

                pred_framewise = list(np.repeat(pred_acts, pred_durs, axis=0))
                target_framewise = list(np.repeat(fut_acts.squeeze(0).detach().tolist(), fut_durs, axis=0))

                if len(pred_framewise) > len(target_framewise):
                    pred_framewise = pred_framewise[:len(target_framewise)]

                elif len(pred_framewise) < len(target_framewise):
                    diff = len(target_framewise) - len(pred_framewise)
                    pred_framewise.extend([pred_framewise[-1]] * diff)

                for pred_per in [.1, .2, .3, .5]:
                    pred_dict[pred_per].extend(pred_framewise[:round(len(pred_framewise) * pred_per) + 1])
                    tar_dict[pred_per].extend(target_framewise[:round(len(target_framewise) * pred_per) + 1])

                    if pred_per == .5:
                        pred = pred_framewise[:round(len(pred_framewise) * pred_per) + 1]
                        tar = target_framewise[:round(len(target_framewise) * pred_per) + 1]

                        pred = [valset.ix2act[ix] for ix in pred]
                        tar = [valset.ix2act[ix] for ix in tar]

                        results = pd.DataFrame({'tar': tar, 'pred': pred})
                        results.to_csv(
                            '50-salads-diagnostics/n_z_{}/results/obs{}/{}.csv'.format(str(n_z), str(obs_per),
                                                                                       str(file_name)))

            for pred_per in [.1, .2, .3, .5]:
                acc = round(balanced_accuracy_score(tar_dict[pred_per], pred_dict[pred_per]), 5)
                acc_dict[obs_per][pred_per].append(acc)

    mean_acc = []

    for obs_per, pred_per_dict in acc_dict.items():
        for pred_per, acc_per_split in pred_per_dict.items():
            mean_acc.append(sum(acc_per_split) / len(acc_per_split))

    all_accs.append(mean_acc)

    results = pd.DataFrame(all_accs, columns=['.2,.1', '.2,.2', '.2,.3', '.2,.5', '.3,.1', '.3,.2', '.3,.3', '.3,.5', ])
    results['mean'] = results.mean(axis=1)
