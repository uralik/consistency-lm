import numpy as np
import torch
import random
import pickle
import os

from torch.utils.data import DataLoader
from pprint import pprint

import self_terminating.base.utils as utils
import self_terminating.base.data as data
from glob import glob


def main(args):
    device = (
        torch.device("cuda:%s" % args.gpu_id)
        if args.gpu_id >= 0
        else torch.device("cpu")
    )

    dirs = []
    if args.sweep_dir is not None:
        if args.dataset_version == "wikitext_sentencized":
            extra_folder_name = "wikitextv2"
            dirs = glob(os.path.join(args.sweep_dir, "*", extra_folder_name))
        else:
            # use glob to get model load directories
            raise NotImplementedError()

    if args.model_load_dir is not None:
        dirs = [args.model_load_dir]

    for d in dirs:
        # Load model and args
        if not args.use_last:
            model_args = pickle.load(open(os.path.join(d, "model_best_args.pkl"), "rb"))
            ckpt = torch.load(os.path.join(d, "model_best.pt"))
        else:
            model_args = pickle.load(open(os.path.join(d, "model_args.pkl"), "rb"))
            ckpt = torch.load(os.path.join(d, "model.pt"))
        vocab = pickle.load(open(os.path.join(d, "model_vocab.pkl"), "rb"))

        nested_args = getattr(model_args, 'model_args', False)
        if nested_args:
            options = nested_args.__dict__
        else:
            options = model_args.__dict__
            nested_args = model_args

        model, criterion, _ = utils.setup_rnn(vocab, **options)
        model.load_state_dict(ckpt["model_dict"])

        args.model_args = nested_args

        if args.dataset_version == "wikitext_sentencized":
            raw_datasets, datasets, _, stats = data.wikitext_sentencized(
                args.dataset_path, model_args.mask_context_k
            )

        datasets["random"] = utils.random_prefixes(
            vocab, model_args.mask_context_k, args.num_random_prefixes
        )

        for ds in args.exclude_datasets:
            try:
                del datasets[ds]
                print(f'{ds} part is excluded from eval')
            except:
                pass

        data_loaders = {
            name: DataLoader(
                datasets[name],
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda x: utils.pad_collate_fn(vocab.get_id("<pad>"), x),
            )
            for name in datasets
        }

        if args.output_dir_override is not None:
            output_dir = args.output_dir_override
        else:
            output_dir = os.path.join(model_args.save_dir, "eval")
        os.makedirs(output_dir, exist_ok=True)

        evaluate(
            output_dir,
            model_args,
            model,
            data_loaders,
            vocab,
            device,
            criterion,
            args.progress_bar,
            args.consistent_sampling,
            args.save_decoding_logs,
        )


def evaluate(
    output_dir,
    model_args,
    model,
    data_loaders,
    vocab,
    device,
    criterion,
    progress_bar=False,
    consistent_sampling=False,
    save_decoding_logs=False,
):
    model.eval()
    ppls = {}
    for data_mode, data_loader_ in data_loaders.items():
        if data_mode == 'random':
            # no targets and ppl for random
            continue
        print(f'Computing PPL for {data_mode}')
        avg_loss_ = utils.compute_ppl_dataloader(
            model_args,
            vocab,
            model,
            criterion,
            data_loader_,
            device,
            True
        )
        ppls[data_mode] = utils.perplexity(avg_loss_)

    
    if isinstance(model, utils.RNNLanguageModelST):
        decoding_algos = ["greedy"]
    else:
        if not args.consistent_sampling:
            decoding_algos = ("beam_2", "beam_4", "greedy", "sample", ("topk", 2), ("topk", 4), ("topp", 0.2),("topp", 0.4))
        else:
            decoding_algos = (("topk", 2), ("topk", 4), ("topp", 0.2),("topp", 0.4))

    decoding_stats = utils.decoding_dataset_stats(
        model,
        data_loaders,
        vocab,
        device,
        num_samples={'train': 2000},
        max_steps=model_args.max_sample_steps,
        temperature=1.0,
        prefix_length=model_args.mask_context_k,
        decoding=decoding_algos,
        progress_bar=progress_bar,
        consistent_sampling=consistent_sampling,
        save_decoding_logs=save_decoding_logs
    )

    decoding_stats['ppl'] = ppls
    with open(os.path.join(output_dir, "eval.pkl"), "wb") as f:
        pickle.dump(decoding_stats, f)

    pprint(decoding_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Either set `--sweep-dir` to a run_slurm dir, or a single `--model-load-dir`
    # The `--model-load-dir` contains files such as `model_best.pt`.
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--output-dir-override", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-last", action='store_true', default=False)

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="training_data/wikitext2-sentencized.json",
    )
    parser.add_argument(
        "--dataset-version", choices=["wikitext_sentencized"], default="wikitext_sentencized"
    )

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--exclude-datasets", default=['valid'], type=str, nargs='+')

    parser.add_argument("--num-random-prefixes", type=int, default=1000)
    parser.add_argument("--num-train-samples", type=int, default=1000)
    parser.add_argument("--max-sample-steps", type=int, default=1500)

    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--progress-bar", action='store_true')

    parser.add_argument("--consistent-sampling", type=int, default=0, choices=[0,1])
    parser.add_argument("--save-decoding-logs", action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
