from enum import Enum, auto

import torch

from model import autoregressive_mask


class DecodingStrategy(Enum):
    """
    Enum class for different decoding strategies.
    """
    TOP_K = auto()
    TOP_P = auto()
    GREEDY = auto()
    RANDOM = auto()
    BEAM_SEARCH = auto()


class SequenceGenerator:
    def __init__(self, model, sos_token, eos_token, pad_token, max_len=50):
        """
        Initializes the sequence generator with a model and parameters for decoding.
        Args:
            model (torch.nn.Module): The trained transformer for generating predictions.
            sos_token (int): The index of the start symbol in the vocabulary.
            eos_token (int): The index of the end symbol in the vocabulary.
            pad_token (int): The index of the padding symbol in the vocabulary.
            max_len (int): The maximum length of the output sequence to generate.
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_len = max_len

    def generate(self, src, src_mask, strategy=DecodingStrategy.GREEDY, k=None, p=None):
        """
        Performs batched autoregressive generation on the model's output using different sampling techniques.
        Args:
            src (torch.Tensor): The encoded source sequence tensor. Shape: [batch_size, seq_len, feature_dim]
            src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [batch_size, 1, seq_len]
            strategy (DecodingStrategy): The decoding strategy to use. Defaults to DecodingStrategy.GREEDY.
        Returns:
            List[List[int]]: A batch of decoded sequences of tokens.
        """
        device = src.device
        B = src.size(0)

        # 1) Encode source once
        if src_mask is not None and src_mask.dim() == 2:
            src_mask = src_mask.unsqueeze(1)      # [B,1,Ls]
        memory = self.model.encode(src, src_mask)

        # 2) Start with SOS
        out_tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)


        for i in range(self.max_len - 1):  # -1 to account for the SOS token
            # YOUR CODE STARTS HERE - do a forward pass through the model to get decoder output
            # Project the decoder output to generator for next token probabilities
            # Hint: use autoregressive_mask to create the tgt_mask
            # Hint: remember that generator gives log_probs, but sampling requires probs
            # TODO: Implement the functionality to get the next token probabilities
            T = out_tokens.size(1)
            pad_mask = (out_tokens != self.pad_token).unsqueeze(1)    # [B,1,T]
            ar_mask = autoregressive_mask(T).to(device)                # [1,T,T]
            tgt_mask = pad_mask.unsqueeze(2) & ar_mask                 # [B,1,T,T]

            dec_out = self.model.decode(memory, src_mask, out_tokens, tgt_mask)   # [B,T,d]
            log_probs = self.model.generator(dec_out[:, -1, :])                    # [B,V] (log)
            probs = torch.exp(log_probs)                                           # [B,V]
            # YOUR CODE ENDS HERE
            # These are the different decoding strategies to generate the next token
            # Will be implemented in the following methods
            if strategy == DecodingStrategy.GREEDY:
                next_word, _ = self.sample_greedy(probs)
            elif strategy == DecodingStrategy.RANDOM:
                next_word, _ = self.sample_random(probs)
            elif strategy == DecodingStrategy.TOP_K:
                next_word, _ = self.sample_top_k(probs, k=k)
            elif strategy == DecodingStrategy.TOP_P:
                next_word, _ = self.sample_top_p(probs, p=p)
            else:
                raise ValueError(f"Invalid decoding strategy: {strategy}")
            # TODO: Implement the functionality to append the next_word to the out_tokens tensor
            # YOUR CODE STARTS HERE
            # Force EOS for already finished rows so sequences stop growing semantically
            next_word = torch.where(finished, torch.full_like(next_word, self.eos_token), next_word)

            # Append and update 'finished'
            out_tokens = torch.cat([out_tokens, next_word.unsqueeze(1)], dim=1)    # [B,T+1]
            finished = finished | (next_word == self.eos_token)
            if finished.all():
                break

        # YOUR CODE ENDS HERE

        # Remove sequences after the end symbol for each batch item
        decoded_sequences = []
        # TODO: Implement the functionality to remove tokens after the EOS token
        # YOUR CODE STARTS HERE
        for b in range(B):
            seq = out_tokens[b].tolist()
            if self.eos_token in seq:
                eos_pos = seq.index(self.eos_token)
                decoded_sequences.append(seq[:eos_pos + 1])
            else:
                decoded_sequences.append(seq)
        # YOUR CODE ENDS HERE
        return decoded_sequences

    def beam_search(self, src, src_mask, beam_width=3):
        """
          Perform beam search decoding for a single input sequence.
          Args:
              src (torch.Tensor): The encoded source sequence tensor. Shape: [1, seq_len, feature_dim]
              src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [1, 1, seq_len]
              beam_width (int): The number of sequences to keep at each step in the beam.
          Returns:
              List[int]: The best sequence of token IDs based on beam search.
      """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search is implemented for a single sequence only."

        device = src.device

        # Encode once (if src is token ids; if it's already encoded, this still works with your model API)
        memory = self.model.encode(src, src_mask)

        # Starting with the initial token.
        ys = torch.full((1, 1), self.sos_token, dtype=torch.long).to(src.device)
        beam_candidates = [(ys, 0)]  # list of tuples (sequence tensor, log probability)


        for _ in range(self.max_len - 1):  # -1 for the sos token
            all_candidates = []
            for ys, log_prob in beam_candidates:
                # TODO: Implement the functionality to get the log probabilities of the next token using the model's decode method
                # YOUR CODE STARTS HERE
                """
                  Steps:
                  1. Get the log probabilities of the next token using the model's decode method.
                  2. Get the top beam_width tokens (by probability values) and their log probabilities.
                  3. Create new candidate sequences by appending each of the top tokens to the current sequence.
                  4. Add the new candidate sequences to the list of all candidates.
                  HINT: The idea will be similar to generate, but you will have to keep track of multiple sequences.
                """
                # If already ended, keep as-is
                if ys[0, -1].item() == self.eos_token:
                    all_candidates.append((ys, log_prob))
                    continue

                # Build tgt mask for this partial sequence
                pad_mask = (ys != self.pad_token).unsqueeze(1).unsqueeze(2)  # [1,1,1,L]
                ar_mask = autoregressive_mask(ys.size(1)).to(device)         # [1,L,L]
                tgt_mask = pad_mask & ar_mask

                # Decode next step
                dec_out = self.model.decode(memory, src_mask, ys, tgt_mask)  # [1,L,d_model]
                log_probs = self.model.generator(dec_out[:, -1, :])           # [1,V] log-softmax

                # Top-k expansions
                top_logp, top_idx = torch.topk(log_probs, k=beam_width, dim=-1)  # [1,k], [1,k]
                for i in range(beam_width):
                    token = top_idx[0, i].unsqueeze(0).unsqueeze(0)               # [1,1]
                    new_seq = torch.cat([ys, token], dim=1)                      # [1,L+1]
                    new_cum = log_prob + top_logp[0, i].item()
                    all_candidates.append((new_seq, new_cum))

            # Keep best beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam_candidates = all_candidates[:beam_width]

            # YOUR CODE ENDS HERE

            # TODO: Implement the functionality to sort all candidates by log probability and select the best beam_width ones
            # YOUR CODE STARTS HERE - Sort all candidates by log probability, select the best beam_width ones
            # Sort all candidates by log probability, select the best beam_width ones

            # YOUR CODE ENDS HERE

            # Check if the end token is generated and stop early
            if all((c[0][0, -1] == self.eos_token) for c in beam_candidates):
                break

        # Choose the sequence with the highest log probability
        best_sequence, _ = max(beam_candidates, key=lambda x: x[1])
        result = best_sequence[0].tolist()
        return result

    @staticmethod
    def sample_greedy(prob):
        """
        Perform greedy decoding to get the next token index based on the probability distribution.
        Steps -
        1. Get the index of the token with the highest probability.
        2. Retrieve the log probability of the chosen token
        Args:
            prob (torch.Tensor): The probability distribution over the target vocabulary of shape
            [batch_size, vocab_size].

        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.gather may be useful.
        """
        # TODO: Implement Greedy Sampling
        # YOUR CODE STARTS HERE
        next_word = torch.argmax(prob, dim=-1)                               # [B]
        chosen = prob.gather(-1, next_word.unsqueeze(1)).squeeze(1)          # [B]
        log_probability_of_next_word = torch.log(chosen.clamp_min(1e-12))    # [B]
        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_random(prob):
        """
        Perform random sampling to get the next token index based on the probability distribution.
        Steps -
        1. Sample from the probability distribution over the target vocabulary.
        2. Retrieve the log probability of the chosen token.
        3. Map sampled indices back to the global vocabulary indices.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.multinomial and torch.gather may be useful.
        """
        # TODO: Implement Random Sampling
        # YOUR CODE STARTS HERE
        next_word = torch.multinomial(prob, num_samples=1).squeeze(1)        # [B]
        chosen = prob.gather(-1, next_word.unsqueeze(1)).squeeze(1)          # [B]
        log_probability_of_next_word = torch.log(chosen.clamp_min(1e-12))    # [B]
        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_top_k(prob, k=5):
        """
        Perform top-k sampling to get the next token index based on the probability distribution.
        Steps -
        1. Filter the top k tokens from the distribution.
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution of top-k tokens to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            k (int): The number of top elements to sample from.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS -
        - The function torch.topk may be useful.
        """
        # TODO: Implement Top-k Sampling
        # YOUR CODE STARTS HERE
        # 1) keep only the top-k per row
        topk_vals, topk_idx = torch.topk(prob, k, dim=-1)                 # [B,k], [B,k]

        # 2) renormalize over the kept items (exact, no epsilon)
        denom = topk_vals.sum(dim=-1, keepdim=True)                       # [B,1]
        topk_probs = topk_vals / denom                                    # [B,k]

        # 3) sample within the top-k (one token per row)
        sampled_in_topk = torch.multinomial(topk_probs, num_samples=1)    # [B,1]

        # 4) map back to original vocab indices
        next_word = topk_idx.gather(1, sampled_in_topk).squeeze(1)        # [B]

        # 5) log prob under the renormalized top-k distribution:
        #    log(topk_vals[i] / sum_j topk_vals[j]) = log(topk_vals[i]) - log(sum)
        chosen_vals = topk_vals.gather(1, sampled_in_topk).squeeze(1)     # [B]
        log_probs = torch.log(chosen_vals) - torch.log(denom.squeeze(1))  # [B]
        # YOUR CODE ENDS HERE
        return next_word, log_probs

    @staticmethod
    def sample_top_p(prob, p=0.9):
        """
        Perform top-p sampling to get the next token index based on the probability distribution.
        Steps -
        1. Retrieve the smallest subset of the distribution that sums just greater than p
        (since = isn't always possible).
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            p (float): The cumulative probability threshold for top-p sampling.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size]
        HINTS:
        - The function torch.cumsum may be useful.
        """
        # TODO: Implement Top-p Sampling
        # YOUR CODE STARTS HERE
        B, V = prob.shape

        # 1) sort descending (keep mapping to original ids)
        sorted_probs, sorted_idx = torch.sort(prob, dim=-1, descending=True)  # [B,V]

        # 2) smallest k with cumsum > p (strictly greater, as per spec)
        csum = torch.cumsum(sorted_probs, dim=-1)                              # [B,V]
        first_over = (csum > p).float().argmax(dim=-1)                         # [B]
        # if never exceeds (e.g., numeric edge cases), keep all
        first_over = torch.where(
            (csum[:, -1] > p),
            first_over,
            prob.new_full((B,), V - 1, dtype=torch.long),
        )

        # mask for kept prefix 0..k
        arange = torch.arange(V, device=prob.device).unsqueeze(0)              # [1,V]
        keep_mask = arange <= first_over.unsqueeze(1)                          # [B,V]

        # 3) renormalize exactly over nucleus (no epsilons)
        kept_vals = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))  # [B,V]
        denom = kept_vals.sum(dim=-1, keepdim=True)                            # [B,1]
        kept_probs = kept_vals / denom                                         # [B,V] (sums to 1 in kept range)

        # 4) sample within nucleus and map back to original vocab ids
        sampled_in_sorted = torch.multinomial(kept_probs, num_samples=1)       # [B,1]
        next_word = sorted_idx.gather(1, sampled_in_sorted).squeeze(1)         # [B]

        # 5) exact log prob: log(chosen / denom) = log(chosen) - log(denom)
        chosen_vals = kept_vals.gather(1, sampled_in_sorted).squeeze(1)        # [B]
        log_probs = torch.log(chosen_vals) - torch.log(denom.squeeze(1))       # [B]
        # YOUR CODE ENDS HERE
        return next_word, log_probs
