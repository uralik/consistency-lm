import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from transformers import top_k_top_p_filtering

import torch.nn.functional as F
import self_terminating.gpt2.train_line as train_utils


def add_args(parser):
    parser.add_argument("--epsilon-upper-bound", type=float, default=0.001)
    return parser


def load_model(args, device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
    if args.model_load_dir:
        model = SelfTerminatingWrapper.from_pretrained(args.model_load_dir, cache_dir=args.cache_dir)
    else:
        model = SelfTerminatingWrapper.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    model.epsilon_upper_bound = args.epsilon_upper_bound
    model.to(device)
    return model, tokenizer


def get_mle_loss(model, batch, pad, eos):
    inp = batch[:, :-1]
    inp_ = inp.clone()
    inp_[inp == pad] = 0

    target = batch[:, 1:]
    target_ = target.clone()
    target_[target == pad] = 0

    inp_mask = inp.ne(pad).float()
    target_mask = target.ne(pad).float()

    model_output = model(inp_, attention_mask=inp_mask)
    logits = model_output[0]
    lprobs = model.st_softmax(
        logits,
        eos
    )
    loss = F.nll_loss(
        lprobs.view(-1, lprobs.size(-1)),
        target_.reshape(-1),
        reduction='none'
    )
    loss = loss * target_mask.view(loss.size())
    loss = loss.sum()
    ntokens = target_mask.sum()
    metrics = train_utils.get_train_metrics(logits, target_, pad)
    metrics['loss_sum'] = loss.item()
    metrics['ntokens'] = ntokens.item()
    loss = loss / ntokens
    return loss, metrics


class SelfTerminatingWrapper(GPT2LMHeadModel):
    """This class supports decoding from a model that uses the self-terminating softmax."""

    def st_softmax(self, logits, eos_token_id, p_eos_prev=None, return_p_eos=False):
        """Self-terminating softmax"""
        left_v_logits = logits[:, :, :eos_token_id]
        right_v_logits = logits[:, :, eos_token_id+1:]
        v_logits = torch.cat([left_v_logits, right_v_logits], dim=-1)
        eos_logits = logits[:, :, eos_token_id].unsqueeze(dim=-1)

        bound = 1 - self.epsilon_upper_bound
        betas = torch.clamp(bound * (torch.sigmoid(eos_logits)), min=1e-20)

        if p_eos_prev is None:
            # Thm 4.1: p = 1 - prod beta
            #       => log(1-p) = sum log beta
            #       => p = 1 - exp sum log beta
            p_eos = 1.0 - torch.exp(torch.cumsum(torch.log(betas), 1))
            p_eos_prev = torch.zeros(p_eos.size(0), 1, 1, device=p_eos.device)
            p_eos_prevs = torch.cat((p_eos_prev, p_eos), 1)[:, :-1, :]
            alphas = torch.clamp(betas * (1 - p_eos_prevs), min=1e-20)

        else:
            alphas = torch.clamp(betas * (1 - p_eos_prev), min=1e-20)
            p_eos = 1 - alphas

        p_Vs = alphas * torch.softmax(v_logits, -1)

        ps = torch.cat((p_Vs, p_eos), dim=2)  # NOTE(assumes EOS is last)
        ps = torch.clamp(ps, min=1e-20)

        ps = ps / ps.sum(-1, keepdim=True)
        log_ps = torch.log(ps)

        if return_p_eos:
            return log_ps, p_eos
        return log_ps

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        # -- begin change ---
        p_eos_prev = None
        # --- end change ---
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            # --- begin change ---
            logits = outputs[0]
            log_ps, p_eos_prev = self.st_softmax(
                logits, eos_token_id,
                p_eos_prev=p_eos_prev,
                return_p_eos=True
            )
            next_token_log_ps = log_ps[:, -1, :]
            p_eos_prev = p_eos_prev[:, -1:, :]
            next_token_logits = next_token_log_ps  # use log_ps as 'logits'
            # --- end change ---
            #next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids