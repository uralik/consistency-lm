import numpy
import logging
import pickle
import random
import os
import torch
from tqdm import tqdm
import tensorboard_logger as logger
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME,
)
from torch.utils.data import Dataset


# -- data utils
class LineDataset(Dataset):
    def __init__(self, split_name, data, pad, total_tokens, token_limit_train, token_limit_eval, context_length):
        self.split_name = split_name
        self.total_tokens = total_tokens
        self.token_limit_train = token_limit_train
        self.token_limit_eval = token_limit_eval
        self.pad = pad
        self.context_length = context_length

        original_len = len(data)
        # NOTE: discard sequences shorter than the context size plus 1
        data = [d for d in data if len(d) > context_length]
        if split_name == 'train':
            if token_limit_train > 0:
                data = [d for d in data if len(d) <= token_limit_train]
        else:
            if token_limit_eval > 0:
                data = [d for d in data if len(d) <= token_limit_eval]

        self.batches = self._make_batches(data)
        print("%s size %d (%d discarded) (max len %d) (%d batches)" % (
            split_name,
            len(data),
            original_len-len(data),
            max(len(d) for d in data),
            len(self.batches)
        ))

    def _make_batches(self, data):
        """Group by similar lengths, then create padded batches that meet the token limit."""
        sorted_data = sorted(data, key=lambda x: -len(x))
        batches = []

        i = 0
        while i < len(sorted_data):
            example = sorted_data[i]

            # The first element will be the longest, which will determine the padded size.
            element_size = len(example)
            batch_size = max(1, self.total_tokens // element_size)

            batch = sorted_data[i:i+batch_size]
            batch = self._pad_batch(batch, element_size)

            batches.append(batch)
            i = i + batch_size

        return batches

    def _pad_batch(self, batch, element_size):
        batch_ = []
        for element in batch:
            element_ = element + [self.pad]*(element_size - len(element))
            assert len(element_) == element_size
            batch_.append(element_)
        return batch_

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index], dtype=torch.long)


def collate_func(batch):
    return batch


def generate_batch(model, tokenizer, batch, context_length, device, max_length):
    with torch.no_grad():
        batch, max_len_in_batch = wrap_context_batch(batch, context_length)

        batch = batch.to(device)
        bpe_prefixes = batch.tolist()
        text_prefixes = [tokenizer.decode(p) for p in bpe_prefixes]
        bpe_decodings = []
        text_decodings = []

        if batch.size(0) > 0:
            outputs = model.generate(
                batch,
                max_length=max(1, max_length),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            full_outputs = outputs.clone()
            for b_ind in range(outputs.size(0)):
                bpe_decoding = outputs[b_ind].tolist()
                if tokenizer.eos_token_id in bpe_decoding:
                    bpe_decoding = bpe_decoding[:bpe_decoding.index(tokenizer.eos_token_id)+1]
                text_decoding = tokenizer.decode(bpe_decoding)

                bpe_decodings.append(bpe_decoding)
                text_decodings.append(text_decoding)

    return bpe_prefixes, text_prefixes, bpe_decodings, text_decodings, full_outputs


def trim(decoded, target, context_length, eos_id):
    prefix = target[:, :context_length].tolist()
    cont_target = target[:, context_length:]
    target_trim = []
    model_trim = []
    if not isinstance(decoded, list):
        decoded = decoded.tolist()
    if not isinstance(cont_target, list):
        cont_target = cont_target.tolist()

    for data_cont, model_cont in zip(cont_target, decoded):
        if eos_id in data_cont:
            data_cont_ = data_cont[:data_cont.index(eos_id)+1]
        else:
            data_cont_ = data_cont
        if eos_id in model_cont:
            model_cont_= model_cont[:model_cont.index(eos_id)+1]
        else:
            model_cont_ = model_cont
        target_trim.append(data_cont_)
        model_trim.append(model_cont_)
    return prefix, target_trim, model_trim


def wrap_context_batch(batch, context_length):
    max_len_in_batch = max([i.numel() for i in batch])
    context_list = []
    for seq in batch:
        if seq.size(0) < context_length:
            continue
        else:
            context_list.append(seq[:context_length])
    if len(context_list) == 0:
        return torch.tensor([], dtype=torch.long), max_len_in_batch
    else:
        return torch.stack(context_list, dim=0), max_len_in_batch


def load_dataset(dataset_path, pad, args):
    raw_dataset_dict = pickle.load(open(dataset_path, "rb"))
    datasets = _load_line_dataset(raw_dataset_dict, pad, args)
    return raw_dataset_dict, datasets


def _load_line_dataset(dataset_dict, pad, args):
    datasets = {}
    for split_name, data in dataset_dict.items():
        datasets[split_name] = LineDataset(
           split_name, data, pad,
            args.chunk_size,
            args.token_limit_train,
            args.token_limit_eval,
            args.context_length
        )
    return datasets


class GPT2Wrapper(GPT2LMHeadModel):

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
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

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


# -- experiment utils
def get_optimizer(model, total_batches, args):
    t_total = total_batches * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler


def load_model(args, device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
    if args.model_load_dir:
        model = GPT2Wrapper.from_pretrained(args.model_load_dir, cache_dir=args.cache_dir)
    else:
        model = GPT2Wrapper.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model.to(device)
    return model, tokenizer


def save(model, save_base_dir):
    print('Saving the model...')
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_base_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_base_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def save_decodings(decodings, args):
    output_dir = os.path.join(args.save_base_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Output directory: %s" % output_dir)
    pickle.dump(decodings, open(os.path.join(output_dir, 'decodings.pkl'), 'wb'))


def save_metrics(metrics, args):
    output_dir = os.path.join(args.save_base_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Output directory: %s" % output_dir)
    pickle.dump(metrics, open(os.path.join(output_dir, 'metrics_best.pkl'), 'wb'))


def _expr_dir(args, include_date=False):
    from datetime import datetime
    now = datetime.now()
    d = os.path.join(
        args.save_base_dir,
        args.expr_name,
        ""
        if not include_date
        else "%d_%d_%d_%d_%d" % (now.year, now.month, now.day, now.hour, now.minute),
    )
    i = 2
    d_ = d
    while os.path.exists(d_):
        d_ = d + "_" + str(i)
        i += 1
    return d_


def setup_tensorboard(args):
    log_directory = args.save_base_dir
    args.log_step = 1
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    try:
        logger.configure(log_directory)
    except ValueError:
        pass


def log_tensorboard(values_dict, step):
    for k, v in values_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            logger.log_value(k, v, step)


def setup(args):
    args.save_base_dir = _expr_dir(args)
    os.makedirs(args.save_base_dir, exist_ok=True)
    setup_tensorboard(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info("device: %s, n_gpu %d".format(str(device), n_gpu))
    logger.info("output directory: %s" % args.save_base_dir)
    return device, logger


