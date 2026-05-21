from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer


def assert_tokenizer_consistency(model_name_1, model_name_2, hf_token=None):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_name_1, token=hf_token).vocab
            == AutoTokenizer.from_pretrained(model_name_2, token=hf_token).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_name_1} and {model_name_2}.")


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        primary_model_name: str,
        primary_model_metrics: list[str],
        primary_model_agg_metrics: Optional[list[str]] = None,
        max_length: int = 512,
        second_model_name: Optional[str] = None,
        second_model_metrics: Optional[list[str]] = None,
        second_model_agg_metrics: Optional[list[str]] = None,
        cross_model_agg_features: Optional[list[str]] = None,
        return_xppl: Optional[bool] = False,
        return_second_model_hs: Optional[bool] = False,
        hidden_state_fusion: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        self.max_length = max_length
        self.primary_model_metrics = primary_model_metrics
        self.primary_model_agg_metrics = primary_model_agg_metrics or []

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
        self.second_model_agg_metrics = second_model_agg_metrics or []
        self.cross_model_agg_features = cross_model_agg_features or []
        self.return_xppl = return_xppl
        self.return_second_model_hs = return_second_model_hs
        self.hidden_state_fusion = hidden_state_fusion or "last"
        self.second_model = None
        if second_model_name is not None:
            assert_tokenizer_consistency(primary_model_name, second_model_name, hf_token=hf_token)
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
        primary_model_hidden_states = self._get_hidden_states(primary_model_outputs.hidden_states)

        primary_model_metrics = self._get_model_metrics(primary_model_logits, encoded_text["input_ids"], self.primary_model_metrics)

        metrics = [primary_model_metrics]
        agg_metrics = []
        if self.primary_model_agg_metrics:
            primary_model_agg_metrics = self._get_model_agg_metrics(
                primary_model_logits,
                encoded_text["input_ids"],
                self.primary_model_agg_metrics,
            )
            agg_metrics.append(primary_model_agg_metrics)

        second_model_hidden_states = None
        if self.second_model is not None:
            with torch.no_grad():
                second_model_outputs = self.second_model(**encoded_text, output_hidden_states=self.return_second_model_hs, use_cache=False,)
            second_model_logits = second_model_outputs.logits
            if self.return_second_model_hs:
                second_model_hidden_states = self._get_hidden_states(second_model_outputs.hidden_states)
            if self.second_model_metrics:
                second_model_metrics = self._get_model_metrics(second_model_logits, encoded_text["input_ids"], self.second_model_metrics)
                metrics.append(second_model_metrics)
            if self.second_model_agg_metrics:
                second_model_agg_metrics = self._get_model_agg_metrics(
                    second_model_logits,
                    encoded_text["input_ids"],
                    self.second_model_agg_metrics,
                )
                agg_metrics.append(second_model_agg_metrics)
            if self.cross_model_agg_features:
                cross_model_agg_features = self._get_cross_model_agg_features(
                    primary_model_logits,
                    second_model_logits,
                    encoded_text["input_ids"],
                    self.cross_model_agg_features,
                )
                agg_metrics.append(cross_model_agg_features)
            if self.return_xppl:
                xppl = self._get_xppl(primary_model_logits, second_model_logits)
                metrics.append(xppl)
        
        metrics = torch.cat(metrics, dim=-1)
        agg_metrics = torch.cat(agg_metrics, dim=-1) if agg_metrics else None

        return {
            "metrics": metrics,
            "agg_metrics": agg_metrics,
            "primary_hidden_states": primary_model_hidden_states,
            "second_hidden_states": second_model_hidden_states,
            "attention_mask": encoded_text["attention_mask"],
        }

    def _get_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.hidden_state_fusion == "uniform":
            layer_hidden_states = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
            normalized_hidden_states = [
                F.layer_norm(hidden_state.float(), hidden_state.shape[-1:])
                for hidden_state in layer_hidden_states
            ]
            return torch.stack(normalized_hidden_states, dim=0).mean(dim=0)

        return hidden_states[-1]
    
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

        shift_logits_model_2 = logits_model_2[:, :-1, :]
        probs_model_2 = torch.softmax(shift_logits_model_2.float(), dim=-1)

        xppl = -(probs_model_2 * log_probs_model_1).sum(dim=-1)

        return xppl.unsqueeze(-1) # [B, T, 1]

    def _get_model_agg_metrics(self, logits: torch.Tensor, input_ids: torch.Tensor, metrics_list: list[str]) -> torch.Tensor:
        eps = torch.finfo(logits.float().dtype).eps
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)

        log_likelihoods = log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1) # [B, T]
        surprisals = -log_likelihoods # [B, T]
        surprisals_diff = torch.diff(surprisals, dim=1) # [B, T-1]
        log_likelihoods_diff_2nd = torch.diff(log_likelihoods, n=2, dim=1) # [B, T-2]

        metrics = []

        if "energy" in metrics_list:
            log_likelihoods_centered = log_likelihoods - log_likelihoods.mean(dim=1, keepdim=True) # [B, T]
            N = log_likelihoods_centered.shape[-1]
            fft_log_probs = torch.fft.fft(log_likelihoods_centered, dim=1) # [B, T]
            power_half = (fft_log_probs.abs() / N).pow(2)[:, :N // 2] # [B, T//2]
            energy_total = -power_half.sum(dim=1) # [B]
            metrics.append(energy_total)

        if "mean" in metrics_list:
            mean = surprisals.mean(dim=1)
            metrics.append(mean)

        if "std" in metrics_list:
            std = surprisals.std(dim=1, correction=0)
            metrics.append(std)
        
        if "var" in metrics_list:
            var = surprisals.var(dim=1, correction=0)
            metrics.append(var)

        if "skew" in metrics_list:
            mean = surprisals.mean(dim=1, keepdim=True)
            diffs = surprisals - mean
            std = surprisals.std(dim=1, keepdim=True, correction=0)
            zscores = diffs / std.clamp_min(eps)
            skew = zscores.pow(3).mean(dim=1)
            metrics.append(skew)

        if "kurtosis" in metrics_list:
            mean = surprisals.mean(dim=1, keepdim=True)
            diffs = surprisals - mean
            std = surprisals.std(dim=1, keepdim=True, correction=0)
            zscores = diffs / std.clamp_min(eps)
            kurtosis = zscores.pow(4).mean(dim=1) - 3.0
            metrics.append(kurtosis)

        if "mean_diff" in metrics_list:
            mean_diff = surprisals_diff.mean(dim=1)
            metrics.append(mean_diff)

        if "std_diff" in metrics_list:
            std_diff = surprisals_diff.std(dim=1, correction=0)
            metrics.append(std_diff)

        if "var_2nd" in metrics_list:
            var_2nd = log_likelihoods_diff_2nd.var(dim=1, correction=0)
            metrics.append(var_2nd)

        if "entropy_2nd" in metrics_list:
            entropies = []
            for sample in log_likelihoods_diff_2nd:
                if sample.numel() == 0:
                    entropies.append(sample.new_zeros(()))
                    continue

                hist = torch.histogram(sample.float(), bins=20, density=False).hist
                probs = hist / hist.sum().clamp_min(eps)
                entropy_2nd = -(probs * probs.clamp_min(eps).log()).sum()
                entropies.append(entropy_2nd.to(log_likelihoods_diff_2nd.dtype))

            metrics.append(torch.stack(entropies))

        if "autocorr_2nd" in metrics_list:
            B, N = log_likelihoods_diff_2nd.shape
            if N > 1:
                shift_1 = log_likelihoods_diff_2nd[:, :-1]
                shift_2 = log_likelihoods_diff_2nd[:, 1:]
                shift_1 = shift_1 - shift_1.mean(dim=1, keepdim=True)
                shift_2 = shift_2 - shift_2.mean(dim=1, keepdim=True)
                numerator = (shift_1 * shift_2).mean(dim=1)
                denominator = shift_1.std(dim=1, correction=0) * shift_2.std(dim=1, correction=0)
                autocorr_2nd = numerator / denominator.clamp_min(eps)
                metrics.append(autocorr_2nd)
            else:
                autocorr_2nd = torch.zeros(B, device=log_likelihoods_diff_2nd.device)
                metrics.append(autocorr_2nd)

        return torch.stack(metrics, dim=-1) # [B, M]
    
    def _get_cross_model_agg_features(
            self,
            logits_model_1: torch.Tensor,
            logits_model_2: torch.Tensor,
            input_ids: torch.Tensor,
            metrics_list: list[str],
        ) -> torch.Tensor:
        eps = torch.finfo(logits_model_1.float().dtype).eps
        shift_logits_model_1 = logits_model_1[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        log_probs_model_1 = torch.log_softmax(shift_logits_model_1.float(), dim=-1)
        log_likelihoods_model_1 = log_probs_model_1.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
        surprisals_model_1 = -log_likelihoods_model_1

        shift_logits_model_2 = logits_model_2[:, :-1, :]
        log_probs_model_2 = torch.log_softmax(shift_logits_model_2.float(), dim=-1)
        log_likelihoods_model_2 = log_probs_model_2.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
        surprisals_model_2 = -log_likelihoods_model_2

        metrics = []

        mean_model_1 = surprisals_model_1.mean(dim=1, keepdim=True)
        mean_model_2 = surprisals_model_2.mean(dim=1, keepdim=True)

        diff_model_1 = surprisals_model_1 - mean_model_1
        diff_model_2 = surprisals_model_2 - mean_model_2

        if "cov" in metrics_list:
            cov = (diff_model_1 * diff_model_2).mean(dim=1)
            metrics.append(cov)

        if "corr" in metrics_list:
            var_model_1 = diff_model_1.pow(2).sum(dim=1)
            var_model_2 = diff_model_2.pow(2).sum(dim=1)
            numerator = (diff_model_1 * diff_model_2).sum(dim=1)
            denominator = torch.sqrt(var_model_1 * var_model_2)
            corr = numerator / denominator.clamp_min(eps)
            metrics.append(corr)

        if "cos_sim" in metrics_list:
            cos_sim = torch.cosine_similarity(surprisals_model_1, surprisals_model_2, dim=1, eps=eps)
            metrics.append(cos_sim)

        return torch.stack(metrics, dim=-1) # [B, M]
