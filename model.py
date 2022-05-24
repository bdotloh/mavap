import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps  # numerical logs


class AnticipationModel(nn.Module):
    def __init__(self, act_dim=20, h_dim=64, z_dim=64, n_z=5):
        super().__init__()

        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_z = n_z

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

        for i in range(n_z):
            post_network = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )

            prior_network = nn.Sequential(
                nn.Linear(h_dim, z_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )

            self.post_networks.append(post_network)
            self.prior_networks.append(prior_network)

        self.phi_z_hidden_to_dec = nn.Linear(h_dim + h_dim, h_dim)
        self.dec_to_act = nn.Linear(h_dim, act_dim)
        self.dur_decoder = nn.Linear(h_dim + h_dim, 1)

        self.rnn = nn.GRUCell(h_dim + h_dim, h_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.gamma_post = nn.Parameter(torch.ones(1, n_z))
        self.gamma_prior = nn.Parameter(torch.ones(1, n_z))

    def forward(self, acts, durs):
        if isinstance(durs, list):
            durs = torch.tensor(durs).unsqueeze(0).type(torch.float)
        if isinstance(acts, list):
            acts = torch.tensor(acts, dtype=torch.long)

        batch_size = acts.size(0)
        hidden = self.init_hidden(acts.size(0))

        T = acts.size(-1)
        act_one_hot = F.one_hot(acts, num_classes=self.act_dim)  # batch * T * act_dim

        kld = 0
        for t in range(T):
            phi_act_t = self.act_emb(act_one_hot[:, t, :].float())
            phi_dur_t = self.dur_emb(durs[:, t].unsqueeze(-1))
            phi_x_t = self.x_emb(torch.cat([phi_act_t, phi_dur_t], -1))

            post_mus = torch.zeros(batch_size, self.n_z, self.z_dim)
            post_sigmas = torch.zeros(batch_size, self.n_z, self.z_dim)

            for i in range(self.n_z):
                post_mus[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden], -1))
                post_sigmas[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden], -1))

            post_mu_t = self.relu(torch.matmul(self.gamma_post, post_mus))
            post_sigma_t = self.softplus(torch.matmul(self.gamma_post, post_sigmas))

            mvn_post_t = Independent(Normal(post_mu_t, post_sigma_t), 1)

            prior_mus = torch.zeros(batch_size, self.n_z, self.z_dim)
            prior_sigmas = torch.zeros(batch_size, self.n_z, self.z_dim)

            for i in range(self.n_z):
                prior_mus[:, i, :] = self.prior_networks[i](hidden)
                prior_sigmas[:, i, :] = self.prior_networks[i](hidden)

            prior_mean_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
            prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

            mvn_prior_t = Independent(Normal(prior_mean_t, prior_sigma_t), 1)

            # kld += self._kld_gauss(post_mu_t, post_scale_t, prior_mean_t, prior_std_t)
            kld += kl_divergence(mvn_post_t, mvn_prior_t)

            z_t = torch.mean(mvn_post_t.rsample((1000,)), 0).squeeze(1)

            hidden = self.rnn(torch.cat([phi_x_t, self.z_emb(z_t)], -1), hidden)

        prior_mus = torch.zeros(batch_size, self.n_z, self.z_dim)
        prior_sigmas = torch.zeros(batch_size, self.n_z, self.z_dim)

        for i in range(self.n_z):
            prior_mus[:, i, :] = self.prior_networks[i](hidden)
            prior_sigmas[:, i, :] = self.prior_networks[i](hidden)

        prior_mean_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
        prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

        mvn_prior_t = Independent(Normal(prior_mean_t, prior_sigma_t), 1)
        z_t = torch.mean(mvn_prior_t.rsample((1000,)), 0).squeeze(1)

        dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([self.z_emb(z_t), hidden], -1)))

        unnorm_prob_pred_act = self.dec_to_act(dec_t)

        dur_dec = self.dur_decoder(torch.cat([self.act_emb(unnorm_prob_pred_act), dec_t], -1))

        pred_dur_mean = dur_dec
        pred_dur_std = self.softplus(dur_dec)

        return unnorm_prob_pred_act, (pred_dur_mean, pred_dur_std), torch.mean(kld)

    def generate(self, acts, durs, total_dur='dur', mean='mean_dur', std='std_dur'):
        if isinstance(durs, list):
            durs = torch.tensor(durs).type(torch.float)
        if isinstance(acts, list):
            acts = torch.tensor(acts, dtype=torch.long)

        last_obs_dur = durs[-1].item()
        batch_size = acts.size(0)
        pred_acts = [acts[:, -1].item()]
        pred_durations = []
        pred_duration_so_far = 0
        with torch.no_grad():
            durs = torch.tensor([(dur - mean) / std for dur in durs]).unsqueeze(0)

            hidden = self.init_hidden(acts.size(0))
            T = acts.size(-1)
            act_one_hot = F.one_hot(acts, num_classes=self.act_dim)

            for t in range(T):
                phi_act_t = self.act_emb(act_one_hot[:, t, :].float())
                phi_dur_t = self.dur_emb(durs[:, t].float()).unsqueeze(0)
                phi_x_t = self.x_emb(torch.cat([phi_act_t, phi_dur_t], -1))

                post_mus = torch.zeros(batch_size, self.n_z, self.z_dim)
                post_sigmas = torch.zeros(batch_size, self.n_z, self.z_dim)

                for i in range(self.n_z):
                    post_mus[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden], -1))
                    post_sigmas[:, i, :] = self.post_networks[i](torch.cat([phi_x_t, hidden], -1))

                post_mu_t = self.relu(torch.matmul(self.gamma_post, post_mus))
                post_sigma_t = self.softplus(torch.matmul(self.gamma_post, post_sigmas))

                mvn_post_t = Independent(Normal(post_mu_t, post_sigma_t), 1)
                z_t = torch.mean(mvn_post_t.rsample((1000,)), 0).squeeze(1)
                phi_z_t = self.z_emb(z_t)

                hidden = self.rnn(torch.cat([phi_x_t, phi_z_t], -1), hidden)

            num_pred = 0

            while True:
                prior_mus = torch.zeros(batch_size, self.n_z, self.z_dim)
                prior_sigmas = torch.zeros(batch_size, self.n_z, self.z_dim)

                for i in range(self.n_z):
                    prior_mus[:, i, :] = self.prior_networks[i](hidden)
                    prior_sigmas[:, i, :] = self.prior_networks[i](hidden)

                prior_mean_t = self.relu(torch.matmul(self.gamma_prior, prior_mus))
                prior_sigma_t = self.softplus(torch.matmul(self.gamma_prior, prior_sigmas))

                mvn_prior_t = Independent(Normal(prior_mean_t, prior_sigma_t), 1)
                z_t = torch.mean(mvn_prior_t.rsample((1000,)), 0).squeeze(1)
                phi_z_t = self.z_emb(z_t)

                dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, hidden], -1)))
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

                hidden = self.rnn(torch.cat([phi_x_t, phi_z_t], -1), hidden)

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

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.h_dim)


class VRNN(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers, bias=False, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # feature-extracting transformations
        self.phi_act = nn.Sequential(
            nn.Linear(act_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim))

        self.phi_x = nn.Sequential(
            nn.Linear(h_dim + 1, h_dim),
            nn.ReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

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

    def forward(self, act_seq, dur_seq):
        if self.batch_first:
            batch_size = act_seq.size(0)
            T = act_seq.size(1)
        else:
            batch_size = act_seq.size(1)
            T = act_seq.size(0)

        all_enc_mean, all_enc_std = [], []
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
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], -1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_act_t = self.dec_act(dec_t)

            dec_dur_mean_t = self.dec_dur_mean(torch.cat([self.phi_act(dec_act_t), dec_t], -1))
            dec_dur_std_t = self.dec_dur_std(torch.cat([self.phi_act(dec_act_t), dec_t], -1))

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_dec_act.append(dec_act_t)
            all_dec_dur_mean.append(dec_dur_mean_t)
            all_dec_dur_std.append(dec_dur_std_t)

        priors_mean = torch.stack(all_prior_mean, dim=1)
        priors_std = torch.stack(all_prior_std, dim=1)

        posteriors_mean = torch.stack(all_enc_mean, dim=1)
        posteriors_std = torch.stack(all_enc_std, dim=1)

        reconstructed_acts = torch.stack(all_dec_act, dim=1)
        reconstructed_durs_mean = torch.stack(all_dec_dur_mean, dim=1)
        reconstructed_durs_std = torch.stack(all_dec_dur_std, dim=1)

        return (priors_mean, priors_std), \
               (posteriors_mean, posteriors_std), \
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


