# CSCI 682
# Transformer Architecture Skeleton

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
  """One causal self-attention head."""

  #n_embed - int - the input embedding dimension
  #head_size - int - the dimension used by this attention head
  #context_size - int - the maximum context length
  def __init__(self, n_embed: int, head_size: int, context_size: int) -> None:
    super().__init__()
    # Each projection maps token representations to one attention head space

    # the nn.Linear() created, 
    # head_size x n_embed sized, matrix
    # is a weight matrix
    #
    # "What do I contain?"
    self.key = nn.Linear(n_embed, head_size, bias=False)
    # "What am I looking for?"
    self.query = nn.Linear(n_embed, head_size, bias=False)
    # "What could I tell *someone* if they attend to me?"
    self.value = nn.Linear(n_embed, head_size, bias=False)
  
    # ones() creates square matrix of all 1's
    # then tril cuts across the diagonal and makes everything above it and to the right 0's
    self.register_buffer("causal_mask", torch.tril(torch.ones(context_size, context_size)))
    # makes the mask 
    self.causal_mask: torch.Tensor

  #x - tensor - input representations with shape B x T x n_embed
  #return - tensor - transformed representations with shape B x T x head_size
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape

    # k, q, and v should each have shape B x T x head_size
    k: torch.Tensor = self.key(x)
    q: torch.Tensor = self.query(x)
    v: torch.Tensor = self.value(x)

    # using negative values to query for the shape values
    # for k is easier because you can know you're getting
    # the last value and then the second to last value and
    # swapping them
    #
    # as we want in the theory for the algorithm, we
    # go ahead and take the head_size value from the shape
    # of k and take the inverse sqrt of it by taking the negative
    # exponent to 1/2
    scores: torch.Tensor = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

    # Mask future positions so token t cannot attend to tokens after t
    scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))

    # Turn masked scores into attention weights
    weights = F.softmax(scores, dim=-1)

    # Return a weighted sum of values, the shape should be B x T x head_size
    return weights @ v

class MultiHeadAttention(nn.Module):
  """Several self-attention heads run in parallel."""

  #n_embed - int - the input and output embedding dimension
  #n_head - int - the number of attention heads
  #context_size - int - the maximum context length
  def __init__(self, n_embed: int, n_head: int, context_size: int) -> None:
    super().__init__()
    assert n_embed % n_head == 0
    head_size: int = n_embed // n_head

    # creates each of the heads for pytorch
    # each head has a random key/query/value given
    self.heads = nn.ModuleList(SelfAttentionHead(n_embed, head_size, context_size) for _ in range(n_head))

    # random weight matrix
    # to use later
    self.proj = nn.Linear(n_embed, n_embed)

  #x - tensor - input representations with shape B x T x n_embed
  #return - tensor - transformed representations with shape B x T x n_embed
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Each head returns B x T x head_size
    head_outputs: list[torch.Tensor] = [head(x) for head in self.heads]
    
    x = torch.cat(head_outputs, dim=-1)
    return self.proj(x)

class FeedForward(nn.Module):
  """A position-wise neural network applied after attention."""

  #n_embed - int - the input and output embedding dimension
  def __init__(self, n_embed: int) -> None:
    super().__init__()

    self.net = nn.Sequential(
      # multiplying the embeds in both the layers
      # sets the width of the hidden layer
      # which essentially "expands the space"
      # which makes it easier to find patters
      nn.Linear(n_embed, 4 * n_embed), # Layer 1
      nn.ReLU(), # the non-linear activation function
      nn.Linear(4 * n_embed, n_embed), # Layer 2
    )

  #x - tensor - input representations with shape B x T x n_embed
  #return - tensor - transformed representations with shape B x T x n_embed
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

class TransformerBlock(nn.Module):
  """A Transformer block with residual connections and layer normalization."""

  #n_embed - int - the input and output embedding dimension
  #n_head - int - the number of attention heads
  #context_size - int - the maximum context length
  def __init__(self, n_embed: int, n_head: int, context_size: int) -> None:
    super().__init__()
    self.attention = MultiHeadAttention(n_embed, n_head, context_size)
    self.feed_forward = FeedForward(n_embed)

    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  #x - tensor - input representations with shape B x T x n_embed
  #return - tensor - transformed representations with shape B x T x n_embed
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attention(self.ln1(x))
    x = x + self.feed_forward(self.ln2(x))
    return x


class TransformerLM(nn.Module):
  """A small decoder-only Transformer language model."""

  #vocab_size - int - the number of tokens in the vocabulary
  #context_size - int - the maximum context length
  #n_embed - int - the token and position embedding dimension
  #n_head - int - the number of attention heads in each block
  #n_layer - int - the number of Transformer blocks
  def __init__(self, vocab_size: int, context_size: int, n_embed: int = 64, n_head: int = 4, n_layer: int = 2) -> None:
    super().__init__()
    self.context_size = context_size

    # encodes what is token is
    # give the token ids 
    # a row of values
    self.token_embedding = nn.Embedding(vocab_size, n_embed)

    # relates positions to the tokens
    # during attention it only looks
    # at combinations; which is why
    # this is required for permuting
    # to get the final output.
    self.position_embedding = nn.Embedding(context_size, n_embed)

    # blocks let tokens gather information from the
    # attention from earlier tokens and then uses that
    # in its feedforward (intra-token consideration)
    self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head, context_size) for _ in range(n_layer)])

    # normalizes the values for comparison
    self.ln = nn.LayerNorm(n_embed)
    # maps each position vector to a score
    # for every token. each logit at a position
    # is the prediction for the next token position
    self.lm_head = nn.Linear(n_embed, vocab_size)

  #idx - tensor - token ids with shape B x T
  #targets - tensor or None - next token ids with shape B x T
  #return - (tensor, tensor or None) - logits and optional loss
  def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T = idx.shape

    token_embeddings = self.token_embedding(idx)
    position_ids = torch.arange(T, device=idx.device)
    position_embeddings = self.position_embedding(position_ids)

    x = token_embeddings + position_embeddings
    x = self.blocks(x)
    x = self.ln(x)

    logits: torch.Tensor = self.lm_head(x)

    loss: torch.Tensor | None = None
    if targets is not None:
      B, T, C = logits.shape

      # .view() changes the
      # matrices into a more
      # workable format for
      # .cross_entropy() to work
      # with.
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)

      # this applies a softmax to the logits
      # then it takes the probability the model
      # gave the correct token. Then computes
      # the negative log of the probability.
      #
      # That negative log more gradually
      # and accurately accounts for differences
      # from 0. to 1.0 that the logits hold
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  @torch.no_grad()
  #idx - tensor - starting token ids with shape B x T
  #max_new_tokens - int - the number of tokens to sample
  #return - tensor - token ids with the sampled tokens appended
  def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    # Here, we will generate a fixed number of tokens
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.context_size :]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]

      # softmax gives you probabilities that sum
      # to 1 across the vocabulary that will be
      # pulled from to make a prediction
      # 
      # `dim=-1` is the vocabulary size
      probabilities = F.softmax(logits, dim=-1)
      # This uses the probabilites to choose what the
      # next word will be by randomly choosing within the
      # probabilties.
      #
      # Doesn't just pick the most likely because that won't
      # always be the correct next word in every situation.
      #
      # `num_samples=1` tells the method to to give you 1
      # possible next token.
      next_idx = torch.multinomial(probabilities, num_samples=1)

      idx = torch.cat((idx, next_idx), dim=1)
    return idx
