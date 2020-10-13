import numpy as np
import self_terminating.base.utils as utils
import pickle
import json
import os


def wikitext_sentencized(path, min_context_length=0):
    raw_datasets = json.load(open(path, 'r'))
    for key in raw_datasets:
        raw_datasets[key] = [x.split() for x in raw_datasets[key]]
    if os.path.exists(path + '.vocab.p'):
        vocab = pickle.load(open(path+'.vocab.p', 'rb'))
    else:
        vocab = utils.Dictionary(raw_datasets, include_valid=True)
        pickle.dump(vocab, open(path+'.vocab.p', 'wb'))

    tokenized_datasets = utils.tokenize_dataset(raw_datasets, vocab, min_context_length=min_context_length)
    datasets = {name: utils.LMDataset(ds) for name, ds in tokenized_datasets.items()}
    stats = {'path': path,
             'num_train': len(raw_datasets['train']),
             'num_valid': len(raw_datasets['valid']),
             'vocab_size': len(vocab),
             'avg_len': np.mean([len(d) for d in raw_datasets['train']])}
    print("Dataset loaded.\n\tTrain size: %d\tValid size: %d\n\t|V|: %d\tmax len: %d\tavg len: %d\n" % (
            len(raw_datasets['train']),
            len(raw_datasets['valid']),
            len(vocab),
            max(len(x) for x in raw_datasets['train']),
            stats['avg_len']))
    return raw_datasets, datasets, vocab, stats

