from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import os

from datautils import build_vocab, read_pairs, create_dataloaders, TranslationDataset
from model_training import TrainingHyperParams, load_trained_model
from sequence_generator import SequenceGenerator, DecodingStrategy
from datautils import convert_and_join, get_special_tokens, BOS_WORD, EOS_WORD, BLANK_WORD


def get_bleu_score(model_path, device, decoding_strategy=DecodingStrategy.GREEDY, max_len=1, k=None, p=None,
                   beam_width=None, batch_size=64, validation=True, src_vocab_size=None, tgt_vocab_size=None):
    training_hp = TrainingHyperParams()
    src_path_test = "data/valid.de-en.de" if validation else "data/test.de-en.de"
    tgt_path_test = "data/valid.de-en.en" if validation else None
    raw_data_test = read_pairs(src_path_test, tgt_path_test)
    src_path = "data/train.de-en.de"
    tgt_path = "data/train.de-en.en"
    raw_data = read_pairs(src_path, tgt_path, test_set=~validation)
    src_data = [sent[0] for sent in raw_data]
    tgt_data = [sent[1] for sent in raw_data]
    src_vocab = build_vocab(src_data, lang="de")
    tgt_vocab = build_vocab(tgt_data, lang="en")
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    dataset = TranslationDataset(raw_data_test, src_vocab, tgt_vocab)
    if decoding_strategy == DecodingStrategy.BEAM_SEARCH:
        batch_size = 1
    dataloader = create_dataloaders(
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        max_padding=training_hp.max_padding
    )
    reversed_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    reversed_src_vocab = {v: k for k, v in src_vocab.items()}
    special_tokens = get_special_tokens()
    transformer = load_trained_model(model_path=model_path, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    sg = SequenceGenerator(
        model=transformer,
        sos_token=special_tokens[BOS_WORD],
        eos_token=special_tokens[EOS_WORD],
        pad_token=special_tokens[BLANK_WORD],
        max_len=max_len
    )
    all_predictions = []
    all_references = []
    for i, data in tqdm(enumerate(dataloader), desc='Decoding translations', total=len(dataloader)):
        if decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            pred = sg.beam_search(
                src=data.src,
                src_mask=data.src_mask,
                beam_width=beam_width,
            )
            pred = [pred]
        else:
            pred = sg.generate(
                src=data.src,
                src_mask=data.src_mask,
                strategy=decoding_strategy,
                k=k,
                p=p
            )
        candidate = convert_and_join(reversed_tgt_vocab, pred)
        reference = convert_and_join(reversed_tgt_vocab, data.tgt_y.cpu().numpy().tolist())
        all_predictions.extend(candidate)
        if validation: # if not validation set, we are not sharing references locally
            all_references.extend(reference)
        if decoding_strategy == DecodingStrategy.BEAM_SEARCH and i >= 1000:
            break

    # Calculate BLEU score
    bleu = None
    if validation: # if not validation set, we are not sharing references locally
        bleu = bleu_score(all_predictions, all_references)

    return all_predictions, all_references, bleu


def generate_test_set_predictions(model_path, device, file_name, decoding_strategy=DecodingStrategy.GREEDY, max_len=1,
                                  k=None, p=None, beam_width=None, batch_size=64, src_vocab_size=None, tgt_vocab_size=None):
    preds, refs, _ = get_bleu_score(model_path, device, max_len=max_len, k=k, p=p, beam_width=beam_width,
                                    decoding_strategy=decoding_strategy, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                    batch_size=batch_size, validation=False)

    df = pd.DataFrame(preds, columns=['predicted'])
    path = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f'{file_name}.csv'), index=False)


def bleu_score(reference_sentences, candidate_sentences):
    tokenized_references = [[nltk.word_tokenize(ref)] for ref in reference_sentences]
    tokenized_candidates = [nltk.word_tokenize(candidate) for candidate in candidate_sentences]
    average_bleu_score = corpus_bleu(tokenized_references, tokenized_candidates)
    return average_bleu_score * 100
