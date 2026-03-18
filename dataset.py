import torch
from torch.utils.data import Dataset
class KaggleDataset(Dataset):
    def __init__(self, df, tokenizer, seq_len):
        self.df = df
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos = tokenizer.token_to_id("[SOS]")
        self.eos = tokenizer.token_to_id("[EOS]")
        self.pad = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        context = row["context"]
        target = row["target"]

        enc_tokens = self.tokenizer.encode(context).ids
        dec_tokens = self.tokenizer.encode(target).ids

        enc_pad = self.seq_len - len(enc_tokens) - 2
        dec_pad = self.seq_len - len(dec_tokens) - 1

        if enc_pad < 0 or dec_pad < 0:
            enc_tokens = enc_tokens[:self.seq_len-2]
            dec_tokens = dec_tokens[:self.seq_len-1]
            enc_pad = 0
            dec_pad = 0

        encoder_input = torch.tensor(
            [self.sos] + enc_tokens + [self.eos] + [self.pad]*enc_pad
        )

        decoder_input = torch.tensor(
            [self.sos] + dec_tokens + [self.pad]*dec_pad
        )

        label = torch.tensor(
            dec_tokens + [self.eos] + [self.pad]*dec_pad
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0