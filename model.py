import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps  # numerical logs


class MAVAP(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers, n_heads):
        super().__init__()

        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.act_emb = nn.Sequential(
            nn.Linear(act_dim, h_dim),
            nn.ReLU()
        )

        self.dur_emb = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.ReLU()
        )

        self.x_emb = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )

        self.z_emb = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        # create list of posterior and prior networks
        self.post_networks = []
        self.prior_networks = []

        for i in range(n_heads):
            post_network = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim),
                nn.ReLU()
            )

            prior_network = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim),
                nn.ReLU(),
            )

            self.post_networks.append(post_network)
            self.prior_networks.append(prior_network)

        self.phi_z_hidden_to_dec = nn.Linear(h_dim + h_dim, h_dim)
        self.dec_to_act = nn.Linear(h_dim, act_dim)
        self.dur_decoder = nn.Linear(h_dim + h_dim, 1)

        self.rnn = self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.gamma_post = nn.Parameter(torch.ones(n_heads))
        self.gamma_prior = nn.Parameter(torch.ones(n_heads))

    def forward(self, acts, durs):
        all_post_mu, all_post_sigma = [], []
        all_prior_mu, all_prior_sigma = [], []
        batch_size = acts.size(0)
        T = acts.size(1)
        hidden = torch.zeros(self.n_layers, batch_size, self.h_dim, device=device)
        for t in range(T):
            #print('timestep: ',t)
            phi_act_t = self.act_emb(acts[:, t, :].float())
            phi_dur_t = self.dur_emb(durs[:, t])
            phi_x_t = self.x_emb(torch.cat([phi_act_t, phi_dur_t], -1))

            post_mus = torch.zeros(batch_size, self.n_heads, self.z_dim)
            post_sigmas = torch.zeros(batch_size, self.n_heads, self.z_dim)

            prior_mus = torch.zeros(batch_size, self.n_heads, self.z_dim)
            prior_sigmas = torch.zeros(batch_size, self.n_heads, self.z_dim)

            #print(f'shape \n phi_x_t: {phi_x_t.shape} \n hidden" {hidden[-1].shape}')
            for i in range(self.n_heads):
                post_mus[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden[-1]], -1))
                post_sigmas[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden[-1]], -1))

                prior_mus[:, i, :] = self.prior_networks[i](hidden[-1])
                prior_sigmas[:, i, :] = self.prior_networks[i](hidden[-1])

            post_mu_t = self.relu(torch.matmul(self.gamma_post, post_mus))
            post_sigma_t = self.softplus(torch.matmul(self.gamma_post, post_sigmas))

            prior_mean_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
            prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

            all_post_mu.append(post_mu_t)
            all_post_sigma.append(post_sigma_t)

            all_prior_mu.append(prior_mean_t)
            all_prior_sigma.append(prior_sigma_t)

            z_t = self._reparameterized_sample(post_mu_t, post_sigma_t)
            # kld += self._kld_gauss(post_mu_t, post_scale_t, prior_mean_t, prior_std_t)
            # kld += kl_divergence(mvn_post_t, mvn_prior_t)
            #print(f'shape phi_z_t {self.z_emb(z_t).shape}')
            _, hidden = self.rnn(torch.cat([phi_x_t, self.z_emb(z_t)], -1).unsqueeze(0), hidden)

        prior_mus = torch.zeros(batch_size, self.n_heads, self.z_dim)
        prior_sigmas = torch.zeros(batch_size, self.n_heads, self.z_dim)

        for i in range(self.n_heads):
            prior_mus[:, i, :] = self.prior_networks[i](hidden[-1])
            prior_sigmas[:, i, :] = self.prior_networks[i](hidden[-1])

        prior_mu_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
        prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

        z_t = self._reparameterized_sample(prior_mu_t, prior_sigma_t)

        dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([self.z_emb(z_t), hidden[-1]], -1)))

        pred_act = self.dec_to_act(dec_t)

        dur_dec = self.dur_decoder(torch.cat([self.act_emb(pred_act), dec_t], -1))

        pred_dur_mean = dur_dec
        pred_dur_std = self.softplus(dur_dec)

        return pred_act, (pred_dur_mean, pred_dur_std), \
               (all_prior_mu, all_prior_sigma), (all_post_mu, all_post_sigma)

    def forecast(self, acts, durs, total_dur='dur', mean='mean_dur', std='std_dur'):
        last_obs_dur = durs[-1].item()

        pred_acts = [torch.argmax(acts[:, -1, :]).item()]
        pred_durations = []
        pred_duration_so_far = 0

        hidden = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        T = acts.size(1)

        for t in range(T):
            phi_act_t = self.act_emb(acts[:, t, :].float())
            phi_dur_t = self.dur_emb(durs[t].float()).unsqueeze(0)

            phi_x_t = self.x_emb(torch.cat([phi_act_t, phi_dur_t], -1))

            post_mus = torch.zeros(1, self.n_heads, self.z_dim)
            post_sigmas = torch.zeros(1, self.n_heads, self.z_dim)

            for i in range(self.n_heads):
                post_mus[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden[-1]], -1))
                post_sigmas[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden[-1]], -1))

            post_mu_t = self.relu(torch.matmul(self.gamma_post, post_mus))
            post_sigma_t = self.softplus(torch.matmul(self.gamma_post, post_sigmas))

            z_t = self._reparameterized_sample(post_mu_t, post_sigma_t)
            phi_z_t = self.z_emb(z_t)

            _, hidden = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), hidden)

        num_pred = 0

        while True:
            prior_mus = torch.zeros(1, self.n_heads, self.z_dim)
            prior_sigmas = torch.zeros(1, self.n_heads, self.z_dim)

            for i in range(self.n_heads):
                prior_mus[:, i, :] = self.prior_networks[i](hidden[-1])
                prior_sigmas[:, i, :] = self.prior_networks[i](hidden[-1])

            prior_mean_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
            prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

            z_t = self._reparameterized_sample(prior_mean_t, prior_sigma_t)
            phi_z_t = self.z_emb(z_t)

            dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, hidden[-1]], -1)))
            pred_act_prob = self.dec_to_act(dec_t)
            pred_act = torch.argmax(self.softmax(pred_act_prob))

            pred_dur_mean = self.sigmoid(
                self.dur_decoder(torch.cat([self.act_emb(pred_act_prob), dec_t], -1)))
            pred_dur_std = self.softplus(
                self.dur_decoder(torch.cat([self.act_emb(pred_act_prob), dec_t], -1)))

            pred_dur_norm = torch.mean(Normal(pred_dur_mean, pred_dur_std).sample((1000,)), 0)
            pred_dur = pred_dur_norm.item() * std + mean

            phi_act_t = self.act_emb(pred_act_prob)
            phi_dur_t = self.dur_emb(pred_dur_norm)
            phi_x_t = self.x_emb(torch.cat([phi_act_t, phi_dur_t], -1))

            _, hidden = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), hidden)

            if num_pred == 0:
                pred_durations.append(int(pred_dur) - last_obs_dur)

            else:
                if pred_acts[-1] != pred_act:
                    pred_acts.append(pred_act.item())
                    pred_durations.append(int(pred_dur))

                elif pred_acts[-1] == pred_act:
                    pred_durations[-1] += int(pred_dur)

            total_dur -= pred_dur
            if pred_duration_so_far >= total_dur:
                break

            num_pred += 1

        return pred_acts, pred_durations

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


class MultiHeadVRNN(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers, n_heads, bias=False, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # feature-extracting transformations
        self.phi_act = nn.Sequential(
            nn.Linear(act_dim, h_dim),
            nn.ReLU())

        self.phi_x = nn.Sequential(
            nn.Linear(h_dim + 1, h_dim),
            nn.ReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # create list of posterior and prior networks
        self.post_networks = []
        self.prior_networks = []

        for i in range(n_heads):
            post_network = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )

            prior_network = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )

            self.post_networks.append(post_network)
            self.prior_networks.append(prior_network)

        # weights on post and priors
        self.gamma_post = nn.Parameter(torch.ones(n_heads))
        self.gamma_prior = nn.Parameter(torch.ones(n_heads))

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_act = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            nn.Sigmoid())
        self.dec_dur_mean = nn.Linear(h_dim + h_dim, 1)
        self.dec_dur_std = nn.Sequential(
            nn.Linear(h_dim + h_dim, 1),
            nn.Softplus()
        )

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, act_seq, dur_seq):
        if self.batch_first:
            batch_size = act_seq.size(0)
            T = act_seq.size(1)
        else:
            batch_size = act_seq.size(1)
            T = act_seq.size(0)

        all_post_mean, all_post_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_act, all_dec_dur_mean, all_dec_dur_std = [], [], []

        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=device)
        for t in range(T):
            if self.batch_first:
                phi_act_t = self.phi_act(act_seq[:, t, :])
                phi_x_t = self.phi_x(torch.cat([phi_act_t, dur_seq[:, t, :]], -1))
            else:
                phi_x_t = self.phi_act(act_seq[t])

            # encoder
            posts = {'means': torch.zeros(batch_size, self.n_heads, self.z_dim),
                     'stds': torch.zeros(batch_size, self.n_heads, self.z_dim)}

            priors = {'means': torch.zeros(batch_size, self.n_heads, self.z_dim),
                      'stds': torch.zeros(batch_size, self.n_heads, self.z_dim)}

            for head_i in range(self.n_heads):
                posts['means'][:, head_i, :] = self.post_networks[head_i](torch.cat([phi_x_t, h[-1]], -1))
                posts['stds'][:, head_i, :] = self.post_networks[head_i](torch.cat([phi_x_t, h[-1]], -1))

                priors['means'][:, head_i, :] = self.prior_networks[head_i](h[-1])
                priors['stds'][:, head_i, :] = self.prior_networks[head_i](h[-1])

            post_mean_t = self.relu(torch.matmul(self.gamma_prior, posts['means']))
            post_sigma_t = self.softplus(torch.matmul(self.gamma_prior, posts['stds']))

            prior_mean_t = self.relu(torch.matmul(self.gamma_prior, priors['means']))
            prior_std_t = self.softplus(torch.matmul(self.gamma_prior, priors['stds']))

            # enc_t = self.enc(torch.cat([phi_x_t, h[-1]], -1))
            # enc_mean_t = self.enc_mean(enc_t)
            # enc_std_t = self.enc_std(enc_t)
            #
            # # prior
            # prior_t = self.prior(h[-1])
            # prior_mean_t = self.prior_mean(prior_t)
            # prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(post_mean_t, post_sigma_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_act_t = self.dec_act(dec_t)

            dec_dur_mean_t = self.dec_dur_mean(torch.cat([self.phi_act(dec_act_t), dec_t], -1))
            dec_dur_std_t = self.dec_dur_std(torch.cat([self.phi_act(dec_act_t), dec_t], -1))

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            all_post_mean.append(post_mean_t)
            all_post_std.append(post_sigma_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_dec_act.append(dec_act_t)
            all_dec_dur_mean.append(dec_dur_mean_t)
            all_dec_dur_std.append(dec_dur_std_t)

        prior_means = torch.stack(all_prior_mean, dim=1)
        prior_stds = torch.stack(all_prior_std, dim=1)

        posterior_means = torch.stack(all_post_mean, dim=1)
        posterior_stds = torch.stack(all_post_std, dim=1)

        reconstructed_acts = torch.stack(all_dec_act, dim=1)
        reconstructed_durs_mean = torch.stack(all_dec_dur_mean, dim=1)
        reconstructed_durs_std = torch.stack(all_dec_dur_std, dim=1)

        return (prior_means, prior_stds), \
               (posterior_means, posterior_stds), \
               reconstructed_acts, \
               (reconstructed_durs_mean, reconstructed_durs_std)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.act_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_act(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_act(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta + EPS) + (1 - x) * torch.log(1 - theta - EPS))

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2 * torch.pi) / 2 + (x - mean).pow(2) / (2 * std.pow(2)))

    def _cross_entropy(self, theta, x):
        # cross entropy loss
        cross_entropy_loss = nn.CrossEntropyLoss()
        return cross_entropy_loss(theta, x)



