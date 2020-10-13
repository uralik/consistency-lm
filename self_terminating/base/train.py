import pickle
import os
import time
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
import copy
import numpy as np
import math

import self_terminating.base.utils as utils
import self_terminating.base.data as data


def train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, args):
    stats_cache = defaultdict(list)
    start = time.time()
    best_val_loss = 1e5
    early_stop = 0

    for epoch_number in range(args.max_epochs):
        sum_loss = 0
        non_pad_tokens = 0

        # -- training
        model.train()
        interval_start = time.time()
        for i, (inp, target) in enumerate(data_loaders["train"]):
            optimizer.zero_grad()
            inp = inp.to(device)
            target = target.to(device)
            logits = model(inp)

            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            # optionally we mask loss computed over the context C here
            if args.mask_context_k > 0:
                context_mask = torch.ones(target.size(), device=target.device)
                context_mask[:, : args.mask_context_k] = 0.0
                loss = loss.view(target.size()) * context_mask
                loss = loss.view(-1)

            num_context_tokens = target.size(0) * args.mask_context_k
            current_non_pad_tokens = (
                target.view(-1).ne(vocab.get_id("<pad>")).nonzero().numel()
                - num_context_tokens
            )
            non_pad_tokens += current_non_pad_tokens

            loss = loss.sum()
            sum_loss += loss.item()
            loss = loss / current_non_pad_tokens

            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

            # now loss is being tracked incrementally
            if i % args.log_every == 0:
                avg_time = (time.time() - interval_start) / args.log_every
                interval_start = time.time()
                avg_loss = sum_loss / non_pad_tokens
                print("Step %d\t Train loss %.4f (%.2f s / step)" % (i, avg_loss, avg_time))
                utils.log_tensorboard({"train/loss": avg_loss}, args.log_step)
            args.log_step += 1

        # -- validation
        sum_valid_loss = 0
        non_pad_tokens = 0
        model.eval()
        with torch.no_grad():
            for i, (inp, target) in enumerate(data_loaders["valid"]):
                inp = inp.to(device)
                target = target.to(device)
                logits = model(inp)

                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                # optionally we mask loss computed over the context C here
                if args.mask_context_k > 0:
                    context_mask = torch.ones(target.size(), device=target.device)
                    context_mask[:, : args.mask_context_k] = 0.0
                    loss = loss.view(target.size()) * context_mask
                    loss = loss.view(-1)

                loss = loss.sum()
                sum_valid_loss += loss

                num_context_tokens = target.size(0) * args.mask_context_k
                current_non_pad_tokens = (
                    target.view(-1).ne(vocab.get_id("<pad>")).nonzero().numel()
                    - num_context_tokens
                )
                non_pad_tokens += current_non_pad_tokens

            avg_val_loss = (sum_valid_loss / non_pad_tokens).item()

            decoding_stats = utils.decoding_dataset_stats(
                model,
                {'valid': data_loaders['valid']},
                vocab,
                device,
                num_samples={'valid': args.num_samples},
                max_steps=args.max_sample_steps,
                temperature=args.temperature,
                prefix_length=args.mask_context_k,
                decoding=("greedy",),
                consistent_sampling=False,
            )

            for data_mode_str, decode_dict in decoding_stats.items():
                for decode_mode_str, metric_dict in decode_dict.items():
                    for metric_name, val in metric_dict.items():
                        utils.log_tensorboard({f"{data_mode_str}/{decode_mode_str}/{metric_name}": val}, args.log_step)

            print(
                "Epoch %d complete.\t Val loss %.4f\t PPL %.2f (best %.2f)"
                % (
                    epoch_number,
                    avg_val_loss,
                    utils.perplexity(avg_val_loss),
                    utils.perplexity(best_val_loss),
                )
            )
            for name in decoding_stats:
                decoding_stats_ = decoding_stats[name]
                print(
                    "%s: Pct. non-term greedy %.4E (avg. len %.1f, uniq. %.4f)"
                    % (
                        name,
                        decoding_stats_["greedy"]["nonterminated"],
                        decoding_stats_["greedy"]["avg_len"],
                        decoding_stats_["greedy"]["uniq_nonterminated"],
                    )
                )
                utils.log_tensorboard(
                    {
                        '%s/greedy_%s' % (name, key): decoding_stats_['greedy'][key]
                        for key in ['nonterminated', 'avg_len']
                    },
                    args.log_step
                )
            now = time.time()
            print(
                "Total time %.1fs (%.1f)s/epoch\n"
                % ((now - start), (now - start) / (epoch_number + 1))
            )

        stats_cache["avg_loss"].append(avg_loss)
        stats_cache["avg_val_loss"].append(avg_val_loss)
        stats_cache["decoding"].append(decoding_stats)

        if avg_val_loss < best_val_loss:
            utils.save(
                args, stats_cache, model, vocab, args.save_dir, "model", best=True
            )
            early_stop = 0
            best_val_loss = avg_val_loss
        else:
            early_stop += 1
            scheduler.step()

        utils.save(args, stats_cache, model, vocab, args.save_dir, "model", best=False)
        utils.log_tensorboard(
            {
                "valid/loss": avg_val_loss,
                "valid/ppl": utils.perplexity(avg_val_loss),
                "valid/best_ppl": utils.perplexity(best_val_loss),
                "early_stop": early_stop,
            },
            args.log_step,
        )

        if early_stop >= args.early_stop:
            break

    print("Performing final evaluation...")
    final_eval(args, model, data_loaders, vocab, device)

def final_eval(args, model, data_loaders, vocab, device):
    ckpt = torch.load(os.path.join(args.save_dir, "model_best.pt"))
    model.load_state_dict(ckpt["model_dict"])
    model.eval()

    del data_loaders['valid']

    decoding_stats = utils.decoding_dataset_stats(
        model,
        data_loaders,
        vocab,
        device,
        num_samples={'train': 2000},
        max_steps=args.max_sample_steps,
        temperature=args.temperature,
        prefix_length=args.mask_context_k,
        decoding=("greedy", "sample",),
    )
    with open(os.path.join(args.save_dir, "final_eval.pkl"), "wb") as f:
        pickle.dump(decoding_stats, f)

    pprint(decoding_stats)

def main(args):
    if args.dataset_version == "wikitext_sentencized":
        raw_datasets, datasets, vocab, stats = data.wikitext_sentencized(
            args.dataset_path, args.mask_context_k
        )
        args.dataset_stats = stats
    else:
        raise NotImplementedError(args.dataset_version)

    # Add random contexts for evaluation
    datasets["random"] = utils.random_prefixes(
        vocab, args.mask_context_k, args.num_samples
    )
    data_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: utils.pad_collate_fn(vocab.get_id("<pad>"), x),
        )
        for name in datasets
    }

    if args.model_load_dir is not None:
        print(f"Loading the model from {args.model_load_dir}...")
        model_args = pickle.load(open(os.path.join(args.model_load_dir, "model_best_args.pkl"), "rb"))
        vocab = pickle.load(open(os.path.join(args.model_load_dir, "model_vocab.pkl"), "rb"))
        ckpt = torch.load(os.path.join(args.model_load_dir, "model_best.pt"))

        nested_args = getattr(model_args, 'model_args', False)
        if nested_args:
            options = nested_args.__dict__
        else:
            options = model_args.__dict__
            nested_args = model_args

        model, criterion, optimizer = utils.setup_rnn(vocab, **options)
        model.load_state_dict(ckpt["model_dict"])
        args.model_args = nested_args
    else:
        model, criterion, optimizer = utils.setup_rnn(vocab, **args.__dict__)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_anneal)

    device = (
        torch.device("cuda:%s" % args.gpu_id)
        if args.gpu_id >= 0
        else torch.device("cpu")
    )
    if args.loss_type == "mle":
        train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, args)
    else:
        raise NotImplementedError(args.loss_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-base-dir", type=str, default="output")
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--expr-name", type=str, default="wikitextv2")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="training_data/wikitext2-sentencized.json",
    )
    parser.add_argument(
        "--include-date", action="store_true", help="include date in expr dir"
    )
    parser.add_argument(
        "--dataset-version", choices=["wikitext_sentencized"], default="wikitext_sentencized"
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--rnn-dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="nn.RNN",
        choices=["nn.RNN", "nn.LSTM"],
    )
    parser.add_argument("--tie-weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--clip-grad-norm", type=float, default=-1)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--momentum-sgd", type=float, default=0.99)
    parser.add_argument("--lr-anneal", type=float, default=1.0)

    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # -- type of loss
    parser.add_argument(
        "--loss-type", type=str, choices=["mle"], default="mle"
    )

    # -- context masking
    parser.add_argument(
        "--mask-context-k",
        type=int,
        default=10,
        help="If >0, loss of prefix up to k is masked",
    )

    # -- self-terminating
    parser.add_argument("--self-terminate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--st-epsilon", type=float, default=0.0)

    # -- validation phase
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-sample-steps", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model-load-path", type=str, default=None)

    args = utils.setup_expr(parser.parse_args())

    if args.tie_weights == 1:
        args.embedding_dim = args.hidden_size

    main(args)
