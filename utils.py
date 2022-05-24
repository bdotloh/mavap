import torch


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i - start)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content) - start)

    return label_seq, length_seq


def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict


def len_to_mask(lengths):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    return mask


def pad_and_merge(sequences, max_len=None):
    """Pads and merges unequal length sequences into batch tensor."""
    dims = sequences[0].shape[1]
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, dims)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end, :] = seq[:end,:]
    if len(sequences) == 1:
        padded_seqs = padded_seqs.float()
    return padded_seqs


def kld_gauss(mean_1, std_1, mean_2, std_2, mask=None):
    """Using std to compute KLD"""
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)

    if mask is not None:
        kld_element = kld_element.masked_select(mask)
    return 0.5 * torch.sum(kld_element)

