from typing import Optional

import torch
import torch.nn as nn

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer


def assert_tokenizer_consistency(model_name_1, model_name_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_name_1).vocab
            == AutoTokenizer.from_pretrained(model_name_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_name_1} and {model_name_2}.")


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        primary_model_name: str,
        primary_model_metrics: list[str],
        max_length: int = 512,
        second_model_name: Optional[str] = None,
        second_model_metrics: Optional[list[str]] = None,
        return_xppl: Optional[bool] = False,
        return_second_model_hs: Optional[bool] = False,
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        self.max_length = max_length
        self.primary_model_metrics = primary_model_metrics

        self.primary_model = AutoModelForCausalLM.from_pretrained(primary_model_name, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(primary_model_name, token=hf_token)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad_token or eos_token; set one before training.")
            tokenizer.pad_token = tokenizer.eos_token

        self.primary_model.config.pad_token_id = tokenizer.pad_token_id
        self.primary_model.eval()
        for parameter in self.primary_model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = tokenizer

        self.second_model_metrics = second_model_metrics or []
        self.return_xppl = return_xppl
        self.return_second_model_hs = return_second_model_hs
        self.second_model = None
        if second_model_name is not None:
            assert_tokenizer_consistency(primary_model_name, second_model_name)
            self.second_model = AutoModelForCausalLM.from_pretrained(second_model_name, token=hf_token)
            self.second_model.config.pad_token_id = tokenizer.pad_token_id
            self.second_model.eval()
            for parameter in self.second_model.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.primary_model.eval()
        if self.second_model is not None:
            self.second_model.eval()
        return self
    
    def forward(self, text: list[str]) -> torch.Tensor:
        self.primary_model.eval()
        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.primary_model.device)
        with torch.no_grad():
            primary_model_outputs = self.primary_model(**encoded_text, output_hidden_states=True, use_cache=False,)
        primary_model_logits = primary_model_outputs.logits
        primary_model_last_hidden_states = primary_model_outputs.hidden_states[-1]

        primary_model_metrics = self._get_model_metrics(primary_model_logits, encoded_text["input_ids"], self.primary_model_metrics)

        metrics = [primary_model_metrics]

        second_model_last_hidden_states = None
        if self.second_model is not None:
            with torch.no_grad():
                second_model_outputs = self.second_model(**encoded_text, output_hidden_states=self.return_second_model_hs, use_cache=False,)
            second_model_logits = second_model_outputs.logits
            if self.return_second_model_hs:
                second_model_last_hidden_states = second_model_outputs.hidden_states[-1]
            if self.second_model_metrics:
                second_model_metrics = self._get_model_metrics(second_model_logits, encoded_text["input_ids"], self.second_model_metrics)
                metrics.append(second_model_metrics)
            if self.return_xppl:
                xppl = self._get_xppl(primary_model_logits, second_model_logits)
                metrics.append(xppl)
        
        metrics = torch.cat(metrics, dim=-1)

        return {
            "metrics": metrics,
            "primary_hidden_states": primary_model_last_hidden_states,
            "second_hidden_states": second_model_last_hidden_states,
            "attention_mask": encoded_text["attention_mask"],
        }
    
    def _get_model_metrics(self, logits: torch.Tensor, input_ids: torch.Tensor, metrics_list: list[str]) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        probs = log_probs.exp()

        next_token_log_probs = log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)

        metrics = []

        if "entropy" in metrics_list:
            entropy = -(probs * log_probs).sum(dim=-1)
            metrics.append(entropy)
        if "max_log_probs" in metrics_list:
            max_log_probs = log_probs.amax(dim=-1)
            metrics.append(max_log_probs)
        if "next_token_log_probs" in metrics_list:
            metrics.append(next_token_log_probs)

        if ("rank" in metrics_list) or ("top_p" in metrics_list):
            mask = log_probs >= next_token_log_probs.unsqueeze(-1)
        
        if "rank" in metrics_list:
            rank = mask.float().mean(dim=-1)
            metrics.append(rank)
        if "top_p" in metrics_list:
            top_p = (probs * mask).sum(dim=-1)
            metrics.append(top_p)
        if "fft" in metrics_list:
            fft_log_probs = torch.fft.fft(next_token_log_probs).abs()
            metrics.append(fft_log_probs)
            
        return torch.stack(metrics, dim=-1) # [B, T, M]

    def _get_xppl(self, logits_model_1: torch.Tensor, logits_model_2: torch.Tensor):
        shift_logits_model_1 = logits_model_1[:, :-1, :]
        log_probs_model_1 = torch.log_softmax(shift_logits_model_1.float(), dim=-1)
        probs_model_1 = log_probs_model_1.exp()

        shift_logits_model_2 = logits_model_2[:, :-1, :]
        log_probs_model_2 = torch.log_softmax(shift_logits_model_2.float(), dim=-1)

        xppl = -(probs_model_1 * log_probs_model_2).sum(dim=-1)

        return xppl.unsqueeze(-1) # [B, T, 1]
