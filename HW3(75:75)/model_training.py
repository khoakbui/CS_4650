"""
Model Training 
"""
import os
from dataclasses import dataclass
import torch

from model import make_model
from utils import LabelSmoothing, LossWrapper
from datautils import init_dataloaders, get_vocab_size, get_pad_id

# constants
MODEL_DIR = "model_weights"

# special tokens 
PAD_ID = get_pad_id()


@dataclass
class TrainingHyperParams:
    """ 
    Hyperparameters for model training. 
    Already provided some initial values, feel free to play around with them.
    Input Params: 
        - train_src_path (str): path to training source data. 
        - train_tgt_path (str): path to training target data. 
        - valid_src_path (str): path to validation source data. 
        - valid_tgt_path (str): path to validation target data. 

        - batch_size (int): number of sequences per batch. 
        - lr (float): learning rate for optimizer. 
        - num_epochs (int): number of epochs to train for. 
        - accum_iter (int): number of batches to accumulate gradients over before updating. 
        - max_padding (int): maximum padding length for sequences. 
        - warmup (int): number of steps to warmup learning rate over. 
        - save_every (int): number of epochs to save model weights every. 

        - continue_training (str): path to model weights. If None, initialize new model in `init_training` and `train`. 
    """
    train_src_path: str = os.path.join(os.getcwd(), 'data/train.de-en.de')
    train_tgt_path: str = os.path.join(os.getcwd(), 'data/train.de-en.en')
    valid_src_path: str = os.path.join(os.getcwd(), 'data/valid.de-en.de')
    valid_tgt_path: str = os.path.join(os.getcwd(), 'data/valid.de-en.en')

    batch_size: int = 64 # number of sequences per batch
    lr: float = 0.0001 # learning rate
    num_epochs: int = 18 # number of epochs to train for
    accum_iter: int = 10 # number of batches to accumulate gradients over before updating
    max_padding: int = 128 # maximum padding length for sequences
    warmup: int = 4000 # number of steps to warmup learning rate over
    save_every: int = 1 # number of epochs to save model weights every
    # format for resuming training - os.path.join(os.getcwd(), 'model_weights', 'model_epoch_4.pt')
    # model_epoch_4.pt is the model weights file which you want to resume training from
    # if None, initialize new model in `init_training` and `train` and start training from scratch
    continue_training: str = None 
    # continue_training: str = os.path.join(os.getcwd(), 'model_weights', 'model_epoch_4.pt') # replace starting epoch
    

@dataclass
class ModelHyperParams:
    """
    Hyperparameters for the Transformer model.
    Input Params:
        - n_blocks (int): number of encoder and decoder blocks.
        - d_model (int): dimension of model.
        - d_ff (int): dimension of feedforward layer.
        - num_heads (int): number of attention heads.
        - dropout (float): dropout rate.
    """
    n_blocks: int = 6 # number of encoder and decoder blocks
    d_model: int = 512 # dimension of model
    d_ff: int = 2048 # dimension of feedforward layer
    num_heads: int = 8 # number of attention heads
    dropout: float = 0.1 # dropout rate


def init_training(training_hp: TrainingHyperParams,
                  model_hp: ModelHyperParams,
                  device: torch.device):
    """
    Initialize training loop: - train/valid dataloaders: initialize dataloaders with the hyperparameters given in
    training_hp - model: initialize model with architecture specified in model_hp; send model to device. - criterion:
    initialize LabelSmoothing loss. Pass it through the `LossWrapper` module. - optimizer: initialize Adam optimizer
    with model parameters. Set learning rate to `training_lr` Input params: - training_hp (TraininHyperParams):
    dataclass specifying training configurations, including paths to train and validation set. See doc string for
    TrainingHyperParams - model_hp (ModelHyperParams): dataclass specifying model architecture. See doc string for
    ModelHyperParams - device (torch.device): device used for training. Returns: Tuple[DataLoader, DataLoader,
    nn.Module, Loss, Optimizer]: returns each of the initialized objects described above.
    """
    print("Initializing training...")

    # get dataloaders
    train_dataloader, valid_dataloader, src_vocab, tgt_vocab = init_dataloaders(training_hp, device)
    print(f"Initialized dataloaders. Training source vocab: {len(src_vocab)}, target vocab: {len(tgt_vocab)}")

    print("Initializing new model...")
    model = make_model(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        n_blocks=model_hp.n_blocks,
        d_model=model_hp.d_model,
        d_ff=model_hp.d_ff,
        h=model_hp.num_heads,
        dropout=model_hp.dropout
    )
    # make model
    if training_hp.continue_training:  # load model from checkpoint
        print(f"Loading model from checkpoint: {training_hp.continue_training}")
        weights = torch.load(training_hp.continue_training, map_location=device)
        model.load_state_dict(weights)

    # send model to device
    model.to(device)

    # initialize loss
    label_smoothing = LabelSmoothing(
        vocab_size=len(tgt_vocab),
        padding_idx=PAD_ID,
        smoothing=0.1)
    criterion = LossWrapper(model.generator, label_smoothing)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_hp.lr)

    return train_dataloader, valid_dataloader, model, criterion, optimizer


def run_epoch(data_iter,
              model,
              criterion,
              optimizer=None,
              train_mode=False,
              accum_iter=1
              ):
    """
    Two functionalities:
        - train_mode = True:  train model for a single epoch.
        - train_mode = False: compute validation loss. In this case, the function does *not* compute gradients, and
        so optimizer is not needed.

    Input Params:
        - data_iter (DataLoader): data used to compute the loss for training or validation pass
        - model (nn.Model): model being trained. (`train_mode=False`,)
        - optimizer (optim): optimizer used for model training.
        - train_mode (bool): gradients are computed when set to True. For False, only the forward pass is computed.
        - accum_iter (int): number of batch steps to accumulate gradient before updating.

    Returns:
        - float: loss per token over the entire batch. In other words:
        (sum of losses over all batches) / (total number of tokens in all batches)


    HINTS: Implement the training loop.
    1. Compute backward pass for loss_per_token.
    2. Do gradient accumulation, namely for every `accum_iter` in batch steps, do following:
    2.a. Update gradients
    2.b. Zero out gradients
    2.c. Update accum_step.
    """
    total_tokens = 0
    total_loss = 0
    tokens = 0
    accum_step = 0  # number of times the gradient has been updated in current epoch

    # check if optimizer is passed for training mode
    if train_mode: assert optimizer, "optimizer not passed for training mode!"

    for batch_step, batch in enumerate(data_iter):

        # compute model forward pass
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        # compute loss
        loss_per_token = criterion(out, batch.tgt_y, batch.ntokens)

        # if training, compute backward pass and accumulate gradients
        if train_mode:
            
            # TODO: Implement the backward pass. 
            # Calculate the gradients for every batch, but accumulate them, and update the model weights every `accum_iter` steps.
            # Work around the `accum_iter` parameter to accumulate gradients and update the model weights.
            # This will be evaluated in non-programming section
            # YOUR CODE STARTS HERE
            # zero grads at the START of each accumulation window
            if (batch_step % accum_iter) == 0:
                optimizer.zero_grad(set_to_none=True)

            # scale loss so that accum_iter mini-batches ≈ one normal step
            (loss_per_token / accum_iter).backward()

            # if we've collected accum_iter mini-batches (or it's the last batch), step
            if ((batch_step + 1) % accum_iter) == 0 or (batch_step + 1) == len(data_iter):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional but helpful
                optimizer.step()
                accum_step += 1
            # YOUR CODE ENDS HERE

        # update total loss and tokens
        total_loss += loss_per_token
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # print loss every 128 batches
        if batch_step % 128 == 1 and train_mode:
            lr = optimizer.param_groups[0]["lr"]
            print("Batch Step: %6d | Accumulation Step: %3d | Loss: %6.2f" % (
                batch_step, accum_step, loss_per_token))
            tokens = 0
        del loss_per_token

    return total_loss / total_tokens


def train(training_hp: TrainingHyperParams, model_hp: ModelHyperParams):
    """
    Standard training loop.
    Iterate through each epoch, calling `run_epoch` with the appropriate mode and data iterators.
    Save model weights every `save_every` epochs, and save the final model weights at the end of training.
    Input Params:
        - training_hp (TrainingHyperParams): dataclass specifying training configurations, including paths to train and
        validation set. See doc string for TrainingHyperParams
        - model_hp (ModelHyperParams): dataclass specifying model architecture. See doc string for ModelHyperParams
    Returns: None

    HINTS:
    1. Iterate through each epoch, calling `run_epoch` with the appropriate mode and data iterators.
    2. Save model weights every `save_every` epochs, and save the final model weights at the end of training.
    """
    # get device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # initialize training
    train_dataloader, valid_dataloader, model, criterion, optimizer = init_training(training_hp, model_hp, device)

    print("Training model...")
    start_epoch = 0
    if training_hp.continue_training is not None:
        epoch_number = int(os.path.basename(training_hp.continue_training).split('_epoch_')[1].split('.')[0])
        start_epoch = epoch_number + 1
    os.makedirs(MODEL_DIR, exist_ok=True)
    for epoch in range(start_epoch, training_hp.num_epochs):
        model.train()
        print(f"Epoch {epoch} Training ====", flush=True)
        _ = run_epoch(
            data_iter=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_mode=True,
            accum_iter=training_hp.accum_iter
        )

        print(f"Epoch {epoch} Validation ====", flush=True)
        model.eval()
        with torch.no_grad():
            sloss = run_epoch(
                data_iter=valid_dataloader,
                model=model,
                criterion=criterion,
                train_mode=False
            )
        print(f"Validation Loss: {sloss}")

        # save model 
        if epoch == training_hp.num_epochs - 1:
            path = os.path.join(MODEL_DIR, f"model_epoch_final.pt")
            torch.save(model.state_dict(), path)
        elif epoch % training_hp.save_every == 0:
            path = os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), path)
        if device == 'cuda':
            torch.cuda.empty_cache()
    print("Training complete.")


def load_trained_model(training_hp=None, model_hp=None, model_path=None, src_vocab_size=None, tgt_vocab_size=None):
    """
    Load a trained model from a checkpoint.
    Input Params:
        - training_hp (TrainingHyperParams): dataclass specifying training configurations, including paths to train and validation set. See doc string for TrainingHyperParams
        - model_hp (ModelHyperParams): dataclass specifying model architecture. See doc string for ModelHyperParams
        - model_path (str): path to model weights.
    Returns:
        - nn.Module: model with weights loaded from checkpoint or newly trained if no checkpoint is found.
    """
    # set default hyperparameters if none are provided
    if training_hp is None: training_hp = TrainingHyperParams()
    if model_hp is None: model_hp = ModelHyperParams()

    # get device
    device = torch.device('cuda' if torch.cuda.is_available()
                          else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # if no model path is provided, train model
    print("Expected model path:", model_path)
    if model_path is None or not os.path.exists(model_path):
        print("No checkpoint found. Training model...")
        train(training_hp, model_hp)
        model_path = os.path.join(MODEL_DIR, "model_epoch_final.pt")
    else:
        print(f"Loading model from checkpoint: {model_path}")

    # get vocab size
    if src_vocab_size is None or tgt_vocab_size is None:
        src_vocab_size, tgt_vocab_size = get_vocab_size()
    print(f"Using source vocab size: {src_vocab_size}, target vocab size: {tgt_vocab_size}")

    # make model
    model = make_model(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        n_blocks=model_hp.n_blocks,
        d_model=model_hp.d_model,
        d_ff=model_hp.d_ff,
        h=model_hp.num_heads,
        dropout=model_hp.dropout
    )

    # load model weights
    try:
      model.load_state_dict(torch.load(model_path, map_location=device))
      model.to(device)
      print("Successfully loaded model")
      return model
    except Exception as e:
      print(f"Exception: {e}")
      print(f"Model failed to load from {model_path}. "
            f"Load model manually. Also make sure your vocab is the correct size.")
