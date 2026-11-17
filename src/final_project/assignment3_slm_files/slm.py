# CSCI 682
# Small Language Model Skeleton

from pathlib import Path
import random
# for importing the data for training
import numpy as np
import matplotlib.pyplot as plt
import torch
from final_project.assignment3_slm_files.transformer import TransformerLM
from final_project.assignment3_slm_files.tinyStoriesPipeline import buildTokenMaps, decode, encode, genBatch, splitStories, toyStories, loadStoriesFromJsonl, learnBPE, tokenizeStories
from final_project.morpheme_tokenizer import MorphemeTokenizer

########################
### Data Preparation ###
########################
#stories - list of strings - story text
#numMerges - int - number of BPE merges
#bpeTrainStories - int or None - number of training stories to use for BPE
#trainStoryLimit - int or None - number of training stories to keep
#seed - int - random seed for the train/validation/test split
#return - encoded training data, encoded validation data, and token maps
def prepData(stories: list[str], numMerges: int, bpeTrainStories: int | None, trainStoryLimit: int | None, seed: int) -> tuple[list[list[int]], list[list[int]], dict[str, int], dict[int, str]]:
  trainStories, validationStories, testStories = splitStories(stories, seed=seed)
  if trainStoryLimit is not None: trainStories = trainStories[:trainStoryLimit]

  bpeStories = trainStories
  if bpeTrainStories is not None: bpeStories = trainStories[:bpeTrainStories]

  morphemeTokenizer = MorphemeTokenizer(str(Path(__file__).resolve().parents[3] / "data"))
  trainStories_tokens = morphemeTokenizer.tokenize(trainStories)
  validStories_tokens = morphemeTokenizer.tokenize(validationStories)

  # creates a map from the training tokens
  trainToken_map = buildTokenMaps(trainStories_tokens)

  # encodes each token in the training set
  t_encoded: list[list[int]] = []
  for t_token in trainStories_tokens:
    t_encoded.append(encode(t_token, trainToken_map[0]))

  # encodes each token in the validation set
  v_encoded: list[list[int]] = []
  for v_token in validStories_tokens:
    v_encoded.append(encode(v_token, trainToken_map[0]))
  
  return t_encoded, v_encoded, trainToken_map[0], trainToken_map[1]

#####################################
### Model Training and Evaluation ###
#####################################
@torch.no_grad()
#model - TransformerLM - language model to evaluate
#data - list of lists of ints - encoded token sequences
#batchSize - int - number of examples per batch
#contextSize - int - context length
#evalIters - int - number of batches to average over
#device - torch device - device on which tensors should be stored
#seed - int - random seed for picking the batches
#return - float - average loss over sampled batches
def estimateLoss(model: TransformerLM, data: list[list[int]], batchSize: int, contextSize: int, evalIters: int, device: torch.device, seed: int) -> float:
  # a fresh generator with the same seed on every call, so each
  # evaluation scores the model on the same validation batches
  rng = np.random.default_rng(seed)
  # TransformerLM is a subclass of Torch
  # so the eval method is coming from that.
  model.eval()

  # keeps track of the losses for every
  # run of the model
  losses: list[float] = []
  for _ in range(evalIters):
    Xnp, Ynp = genBatch(data, batchSize, contextSize, rng)
    X = torch.tensor(Xnp, dtype=torch.long, device=device)
    Y = torch.tensor(Ynp, dtype=torch.long, device=device)
    _, loss = model(X, Y)
    losses.append(loss.item())

  model.train()

  return float(np.mean(losses))

#model - TransformerLM - language model to train
#trainData - list of lists of ints - encoded training sequences
#validationData - list of lists of ints - encoded validation sequences
#steps - int - number of parameter updates
#batchSize - int - number of examples per batch
#contextSize - int - context length
#learningRate - float - optimizer learning rate
#device - torch device - device on which tensors should be stored
#evalInterval - int - number of updates between loss estimates
#evalIters - int - number of batches to average for validation loss
#seed - int - random seed for picking the batches
#return - list of dicts - training and validation losses over time
def train(model: TransformerLM, trainData: list[list[int]], validationData: list[list[int]], steps: int, batchSize: int, contextSize: int, learningRate: float, device: torch.device, seed: int, evalInterval: int = 100, evalIters: int = 5) -> list[dict[str, float]]:
  optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
  rng = np.random.default_rng(seed)
  history: list[dict[str, float]] = []

  for step in range(steps):
    Xnp, Ynp = genBatch(trainData, batchSize, contextSize, rng)
    X = torch.tensor(Xnp, dtype=torch.long, device=device)
    Y = torch.tensor(Ynp, dtype=torch.long, device=device)

    _, loss = model(X, Y)

    # clears the left over gradients
    optimizer.zero_grad()
    # backpropagates the loss to update the
    # weights to better conform to what was
    # actually expected
    loss.backward()
    # uses the back prop
    # to actually apply
    # the update to the weights
    optimizer.step()

    if step == 0 or (step + 1) % evalInterval == 0:
      validationLoss = estimateLoss(model, validationData, batchSize, contextSize, evalIters, device, seed)
      
      # step + 1 because it
      # starts at 0
      #
      # loss.item() because loss
      # is a tensor, and by taking
      # item it returns the float value
      history.append({
        "step": step + 1,
        "train_loss": loss.item(),
        "validation_loss": validationLoss,
      })

      print(f"step {step + 1}: train loss {loss.item():.4f}, validation loss {validationLoss:.4f}")

  return history

#history - list of dicts - training and validation losses over time
#outFile - str - file where the plot should be saved
def plotLosses(history: list[dict[str, float]], outFile: str) -> None:
  steps = [item["step"] for item in history]
  trainLosses = [item["train_loss"] for item in history]
  validationLosses = [item["validation_loss"] for item in history]

  plt.figure(figsize=(7, 5))
  plt.plot(steps, trainLosses, marker="o", label="training loss")
  plt.plot(steps, validationLosses, marker="o", label="validation loss")
  plt.xlabel("update step")
  plt.ylabel("loss")
  plt.legend()
  plt.tight_layout()
  plt.savefig(outFile, dpi=200)
  plt.close()

#model - TransformerLM - trained language model
#tokenToID - dict - map from tokens to integer ids
#idToToken - dict - map from integer ids to tokens
#device - torch device - device on which tensors should be stored
#maxNewTokens - int - number of tokens to generate
#return - list of strings - decoded generated tokens
def generateSample(model: TransformerLM, tokenToID: dict[str, int], idToToken: dict[int, str], device: torch.device, maxNewTokens: int = 50) -> list[str]:
  startID = tokenToID["<s>"]
  idx = torch.tensor([[startID]], dtype=torch.long, device=device)

  # for inference
  # when a back prop
  # is not needed after the result
  with torch.no_grad():
    generated = model.generate(idx, max_new_tokens=maxNewTokens)

  # gets the ids that were predicted
  # to come after the given token
  ids = generated[0].tolist()

  # takes the ids and converts them into
  # string tokens to forumlate the text
  return decode(ids, idToToken)

#tokens - list of strings - generated BPE tokens
#return - string - generated text
def formatSample(tokens: list[str]) -> str:
  text = "".join(token for token in tokens if token != "<s>")
  text = text.replace("</w>", "") #remove end of word tokens
  text = text.replace("</s>", "\n") #replace end of sentence tokens with newlines
  return text.strip()

##################
### Shape Test ###
##################
def runShapeTest() -> None:
  torch.manual_seed(123)
  B = 2
  T = 4
  vocabSize = 10
  contextSize = 8
  nEmbed = 8
  nHead = 2
  nLayer = 1
  maxNewTokens = 3

  model = TransformerLM(vocab_size=vocabSize,
                        context_size=contextSize,
                        n_embed=nEmbed,
                        n_head=nHead,
                        n_layer=nLayer)
  idx = torch.randint(0, vocabSize, (B, T))
  targets = torch.randint(0, vocabSize, (B, T))

  tokenEmbeddings = model.token_embedding(idx)
  positionIDs = torch.arange(T)
  positionEmbeddings = model.position_embedding(positionIDs)
  x = tokenEmbeddings + positionEmbeddings
  afterBlocks = model.blocks(x)
  afterLn = model.ln(afterBlocks)
  logits = model.lm_head(afterLn)
  _, loss = model(idx, targets)
  generated = model.generate(idx, max_new_tokens=maxNewTokens)

  print("Transformer shape test")
  print(f"input token ids:        {tuple(idx.shape)}")
  print(f"token embeddings:       {tuple(tokenEmbeddings.shape)}")
  print(f"position ids:           {tuple(positionIDs.shape)}")
  print(f"position embeddings:    {tuple(positionEmbeddings.shape)}")
  print(f"combined embeddings:    {tuple(x.shape)}")
  print(f"after Transformer blocks: {tuple(afterBlocks.shape)}")
  print(f"after final LayerNorm:  {tuple(afterLn.shape)}")
  print(f"logits:                 {tuple(logits.shape)}")
  print(f"loss:                   {tuple(loss.shape)}")
  print(f"generated token ids:    {tuple(generated.shape)}")

############
### Main ###
############
def main() -> None:
  # one seed for every random choice: the story split, model weight
  # initialization, training and validation batches, and sampling
  seed = 123
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)  # also seeds CUDA and MPS

  # GPUs (CUDA, MPS) can add floating point numbers in a different
  # order on each run, so the losses drift apart after a few hundred
  # steps even with the same seed. The CPU gives the exact same
  # numbers every run. Set this to False to train on a GPU instead.
  deterministic = True
  if deterministic:
    device = torch.device("cpu")
    torch.use_deterministic_algorithms(True)
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

  dataPath = str(Path(__file__).resolve().parents[2] / "data" / "tinyStories.jsonl")
  maxStories: int = 1000
  trainStories: int | None = None
  numMerges = 200
  bpeTrainStories: int | None = None
  steps = 1000
  batchSize = 16
  contextSize = 32
  nEmbed = 64
  nHead = 4
  nLayer = 2
  learningRate = 1e-3
  evalInterval = 100
  evalIters = 5
  plotPath = "slm_loss.png"
  maxNewTokens = 50
  numSamples = 5

  # Check tensor shapes through the TransformerLM architecture
  if False: runShapeTest()

  # Load stories, print split sizes, and encode stories as token IDs
  if True:
    stories: list[str] = []

    useToyData = False
    if useToyData:
      stories = toyStories()
      if maxStories is not None: stories = stories[:maxStories]
    else: stories = loadStoriesFromJsonl(dataPath, maxStories=maxStories)

    trainStoriesCount, validationStoriesCount, testStoriesCount = splitStories(stories, seed=seed)

    print(f"training stories: {len(trainStoriesCount)}")
    print(f"validation stories: {len(validationStoriesCount)}")
    print(f"test stories: {len(testStoriesCount)}")

    trainData, validationData, tokenToID, idToToken = prepData(stories,
                                                               numMerges=numMerges,
                                                               bpeTrainStories=bpeTrainStories,
                                                               trainStoryLimit=trainStories,
                                                               seed=seed)

  # Set up the Transformer model
  if True:
    model = TransformerLM(vocab_size=len(tokenToID),
                          context_size=contextSize,
                          n_embed=nEmbed,
                          n_head=nHead,
                          n_layer=nLayer).to(device)

    print(f"device: {device}")
    print(f"vocabulary size: {len(tokenToID)}")
    print(f"encoded training stories: {len(trainData)}")
    print(f"encoded validation stories: {len(validationData)}")

  # Train the model
  if True:
    history = train(model,
                    trainData,
                    validationData,
                    steps=steps,
                    batchSize=batchSize,
                    contextSize=contextSize,
                    learningRate=learningRate,
                    device=device,
                    seed=seed,
                    evalInterval=evalInterval,
                    evalIters=evalIters)

    plotLosses(history, plotPath)

  # Generate samples from the trained model
  if True:
    for i in range(numSamples):
      print(f"sample {i + 1}:")
      sample = generateSample(model,
                              tokenToID,
                              idToToken,
                              device,
                              maxNewTokens=maxNewTokens)
      print(formatSample(sample))


if __name__ == "__main__":
  main()
