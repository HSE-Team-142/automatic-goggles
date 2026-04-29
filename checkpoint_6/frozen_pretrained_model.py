import torch
import torch.nn as nn

import numpy as np

from scipy.stats import skew, kurtosis, entropy as calc_entropy
from scipy.fftpack import fft, ifft

from transformers import AutoModelForCausalLM, AutoTokenizer

class FronzenPretrainedModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int = 512
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad_token or eos_token; set one before training.")
            tokenizer.pad_token = tokenizer.eos_token

        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = tokenizer

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self
    
    def forward(self, text: list[str]) -> torch.Tensor:
        self.model.eval()
        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**encoded_text, output_hidden_states=True)
        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]

        metrics = self._get_metrics(logits, encoded_text["input_ids"])

        return metrics, last_hidden_states, encoded_text["attention_mask"]
    
    def _get_metrics(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        probs = log_probs.exp()

        next_token_log_probs = log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)

        entropy = -(probs * log_probs).sum(dim=-1)
        max_log_probs = log_probs.amax(dim=-1)

        mask = log_probs >= next_token_log_probs.unsqueeze(-1)
        rank = mask.float().mean(dim=-1)
        top_p = (probs * mask).sum(dim=-1)

        # surprisals = -next_token_log_probs

        # s = surprisals.numpy()
        # mean_s, std_s, var_s, skew_s, kurt_s = np.mean(s), np.std(s), np.var(s), skew(s), kurtosis(s)
        # diff_s = np.diff(s)
        # mean_diff, std_diff = np.mean(diff_s), np.std(diff_s)
        # first_order_diff = np.diff(next_token_log_probs)
        # second_order_diff = np.diff(first_order_diff)
        # var_2nd, entropy_2nd = np.var(second_order_diff), calc_entropy(np.histogram(second_order_diff, bins=20, density=True)[0])
        # autocorr_2nd = np.corrcoef(second_order_diff[:-1], second_order_diff[1:])[0, 1] if len(second_order_diff) > 1 else 0

        # N = next_token_log_probs.shape[-1]
        # fft_log_probs = fft(next_token_log_probs.numpy())
        # power_half = (((np.abs(fft_log_probs))/N) ** 2)[:,:(int(N/2))]
        # energy = np.sum(power_half, axis=1)

        return torch.stack([next_token_log_probs, entropy, max_log_probs, rank, top_p], dim=-1)

        # return [
        #     next_token_log_probs,
        #     entropy, 
        #     max_log_probs,
        #     rank,
        #     top_p,
        #     mean_s,
        #     std_s,
        #     var_s,
        #     skew_s,
        #     kurt_s,
        #     mean_diff,
        #     std_diff,
        #     var_2nd,
        #     entropy_2nd,
        #     autocorr_2nd,
        #     energy,
        # ]
    
