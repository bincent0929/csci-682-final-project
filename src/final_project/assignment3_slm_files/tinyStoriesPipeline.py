# CSCI 682
# TinyStories (https://arxiv.org/abs/2305.07759) pipeline skeleton

import collections
import json
import random
import re
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nltk
from datasets import load_dataset
from ftfy import fix_text
from nltk.tokenize import sent_tokenize

#Use the project-local tokenizer data when it has been downloaded there.
localNltkData = Path(__file__).resolve().parents[2] / ".nltk_data"
if localNltkData.is_dir():
  nltk.data.path.insert(0, str(localNltkData))

####################
### Data Loading ###
####################
#Fix encoding artifacts using the ftfy library
#text - str - the text to clean
#return - str - cleaned text
def cleanText(text: str) -> str:
  return fix_text(text)

#Download a prefix of the TinyStories dataset and save it as one JSONL file
#out - str - file to save the JSONL file
#size - int - the number of stories
#split - str - the Hugging Face split to read from
#streaming - bool - if true, stream the data instead of downloading the full split
#return - int - the number of records saved
def downloadTinyStoriesPrefix(out: str, size: int = 10000, split: str = "train", streaming: bool = True) -> int:
  dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)

  count = 0
  with open(out, "w", encoding="utf-8") as file:
    for record in dataset:
      if count >= size: break
      output = {"id": count,
                "text": cleanText(record["text"])}
      file.write(json.dumps(output, ensure_ascii=False) + "\n")
      count += 1
  return count

#Load stories from a JSONL file with one record per story
#path - str - a JSONL file containing records
#maxStories - int or None - if not None, stop after this many stories
#return - list - loaded stories
def loadStoriesFromJsonl(path: str, maxStories: int | None = None) -> list[str]:
  stories: list[str] = []
  with open(path, encoding="utf-8") as file:
    for line_number, line in enumerate(file):
      if maxStories is not None and line_number >= maxStories: break
      record = json.loads(line)
      stories.append(record["text"])
  return stories

##################################
### Splitting and Tokenization ###
##################################
#Randomly split whole stories into training, validation, and test sets
#stories - list - stories to split
#trainFrac - float - fraction of stories to put in the training set
#valFrac - float - fraction of stories to put in the validation set
#seed - int - random seed for shuffling
#return - (list, list, list) - train, validation, and test stories
def splitStories(stories: list[str], trainFrac: float = 0.8, valFrac: float = 0.1, seed: int | None = None) -> tuple[list[str], list[str], list[str]]:
  shuffled = list(stories)
  random.Random(seed).shuffle(shuffled)

  n = len(shuffled)
  trainEnd = round(n * trainFrac)
  valEnd = trainEnd + round(n * valFrac)

  train = shuffled[:trainEnd]
  validation = shuffled[trainEnd:valEnd]
  test = shuffled[valEnd:]
  return train, validation, test

#Split stories into sentences so BPE merges never span a sentence boundary
#stories - list - raw story strings
#return - list - flattened list of sentence strings, in story order
def sentenceSplitStories(stories: list[str]) -> list[str]:
  sentences: list[str] = []
  for story in stories:
    sentences.extend(sent_tokenize(story))
  return sentences

#Tokenize each sentence in each story and add atomic sentence-boundary tokens
#stories - list - raw story strings
#merges - list - ordered BPE merge pairs
#return - list of lists - one token sequence per story
def tokenizeStories(stories: list[str], merges: list[tuple[str, str]]) -> list[list[str]]:
  tokenizedStories: list[list[str]] = []

  for story in stories:
    storyTokens: list[str] = []
    sentences = sent_tokenize(story)
    tokenizedSentences = tokenizeBPE(sentences, merges)

    for sentenceTokens in tokenizedSentences:
      # Add the tags after BPE so they remain complete tokens instead of
      # being split into the characters "<", "s", and ">".
      storyTokens.extend(["<s>", *sentenceTokens, "</s>"])

    tokenizedStories.append(storyTokens)

  return tokenizedStories

#Apply one merge pair to every story's token sequence
#tokenizedStories - list of lists - per-story token sequences
#pair - (str, str) - the token pair to merge
#return - list of lists - token sequences with the pair merged
def applyMerge(tokenizedStories: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
  mergedToken = pair[0] + pair[1]
  result: list[list[str]] = []
  for strings in tokenizedStories:
    new_strings: list[str] = []
    i = 0
    while i < len(strings): # <- makes the swaps
      if i + 1 < len(strings) and pair == (strings[i], strings[i + 1]):
        new_strings.append(mergedToken)
        i += 2
      else:
        new_strings.append(strings[i])
        i += 1
    result.append(new_strings)
  return result

#Learn a byte pair encoding from training stories
#stories - list - training stories (raw strings)
#numMerges - int - number of merge operations to learn
#return - list - ordered list of merged (tokenA, tokenB) pairs
def learnBPE(stories: list[str], numMerges: int) -> list[tuple[str, str]]:
  tokenizedStories: list[list[str]] = [list(story) for story in stories]

  merges: list[tuple[str, str]] = []
  for _ in range(numMerges):
    # need to grab counts again after each merge
    adjacent_tokens_counts: dict[tuple[str, str], int] = {}
    for strings in tokenizedStories: # <- gets all the current tokens
      i = 0
      while i < len(strings) - 1:
        pair = (strings[i], strings[i + 1])
        adjacent_tokens_counts[pair] = adjacent_tokens_counts.get(pair, 0) + 1
        i += 1

    # don't really think this is super needed
    # but I guess if a "story" was only a few
    # words and there were many merges,
    # then it would make sense.
    if not adjacent_tokens_counts: break

    most_com_tok_pair = max(adjacent_tokens_counts, key=lambda pair: adjacent_tokens_counts[pair])
    merges.append(most_com_tok_pair)
    tokenizedStories = applyMerge(tokenizedStories, most_com_tok_pair)

  return merges

#Tokenize stories using a previously learned byte pair encoding
#stories - list - stories (raw strings) to tokenize
#merges - list - ordered list of merged pairs, from learnBPE run
#return - list of lists - tokenized stories
def tokenizeBPE(stories: list[str], merges: list[tuple[str, str]]) -> list[list[str]]:
  tokenizedStories: list[list[str]] = [list(story) for story in stories]
  for pair in merges:
    tokenizedStories = applyMerge(tokenizedStories, pair)
  return tokenizedStories
    

###################################
### BPE Map, Encode, and Decode ###
###################################
#Construct a token to id and an id to token map from a tokenization
#tokens - list of lists - tokenized examples
#return - (dict, dict) - tokenToID and idToToken maps
def buildTokenMaps(tokens: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
  # <unk> stands in for tokens that never appeared
  # in training (e.g. a rare character in validation)
  unique_toks: set[str] = {"<unk>"}
  # goes through each of the lists of tokens
  # and adds them to a list and then sorts them
  # by the string values
  for x in tokens:
    for token in x:
      unique_toks.add(token)
  ordered_toks = sorted(unique_toks)

  # adds an enumeration to the sorted tokens
  # then creates indeces of ID and token relations
  tok_to_id: dict[str, int] = {}
  id_to_tok: dict[int, str] = {}
  for i, token in enumerate(ordered_toks):
    tok_to_id[token] = i
    id_to_tok[i] = token

  return tok_to_id, id_to_tok

#Construct maps for the complete learned BPE vocabulary
#trainingSentences - list - training sentences used to learn BPE
#merges - list - ordered BPE merge pairs
#return - (dict, dict) - tokenToID and idToToken maps
def buildBPETokenMaps(trainingSentences: list[str], merges: list[tuple[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
  vocabulary: set[str] = {"<s>", "</s>"}

  # BPE begins with individual characters.
  for sentence in trainingSentences:
    vocabulary.update(sentence)

  # Every merge creates a learned token. Some intermediate tokens may not
  # remain in the final training tokenization but can occur in new text.
  for first, second in merges:
    vocabulary.add(first + second)

  return buildTokenMaps([list(vocabulary)])

#Convert a sequence of tokens to int IDs
#seq - list - token sequence to encode
#tokenToID - dict - map from tokens to int IDs
#return - list - corresponding int IDs
def encode(seq: list[str], tokenToID: dict[str, int]) -> list[int]:
  ids: list[int] = []
  for token in seq:
    # unseen tokens fall back to <unk> instead of raising a KeyError
    if token in tokenToID: ids.append(tokenToID[token])
    else: ids.append(tokenToID["<unk>"])
  return ids

#Convert a sequence of int IDs back to tokens
#ids - list - int IDs to decode
#idToToken - dict - map from int IDs to tokens
#return - list - corresponding string tokens
def decode(ids: list[int], idToToken: dict[int, str]) -> list[str]:
  tokens: list[str] = []
  for i in ids:
    tokens.append(idToToken[i])
  return tokens

########################
### Batch Generation ###
########################
#Sample a batch of input-target pairs without crossing story boundaries
#data - list of lists - encoded stories
#B - int - batch size
#T - int - context length
#rng - np.random.Generator or None - random number generator
#return - tuple[np.ndarray, np.ndarray] - X and Y arrays, each with shape B by T
def genBatch(data: list[list[int]], B: int, T: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    if B <= 0 or T <= 0:
        raise ValueError("B and T must both be positive")

    # checks to make sure that each story is big enough to fit the context
    validStories = [story for story in data if len(story) >= T + 1]
    if not validStories:
        raise ValueError("No story is long enough for the requested context length")

    # startCounts checks how many indices in that the
    # batcher can go before starting the sampling.
    startCounts = np.array([len(story) - T for story in validStories])
    # this stores the probabilities for which story may be chosen
    storyProbabilities = startCounts / startCounts.sum()

    X = np.zeros((B, T), dtype=np.int64)
    Y = np.zeros((B, T), dtype=np.int64)

    for row in range(B):
        # picks the story to be batched
        storyIndex = rng.choice(len(validStories), p=storyProbabilities)
        story = validStories[storyIndex]
        # picks a random valid start point for the story
        start = rng.integers(0, startCounts[storyIndex])
        # samples from that start point until the context is filled
        X[row] = story[start:start + T]
        # goes back and samples from 1 after the start
        # until context is filled again.
        Y[row] = story[start + 1:start + T + 1]

    return X, Y

############
### Main ###
############
#A small toy corpus generated with ChatGPT for quick debugging
#return - list - toy stories
def toyStories() -> list[str]:
  return ["Once upon a time, a small cat found a red ball. The cat played all day.",
          "A little dog saw the cat. The dog wanted the red ball too.",
          "The quick brown fox jumped over the lazy dog.",
          "A child told a tiny story about kindness and sharing.",
          "Every night, the child read one small story before bed.",
          "The cat and the dog became friends in the sunny yard.",
          "A small bird sang a song. The child smiled at the bird.",
          "The lazy dog slept while the quick fox ran.",
          "A red ball rolled under the bed, and the cat followed it.",
          "Kind friends shared toys and told stories together."]


#Run the full preprocessing pipeline
def main() -> None:
  useToyData = True
  downloadData = False
  maxStories: int | None = None
  numMerges = 200
  splitSeed = 42
  batchSeed = 42
  dataPath = "tinyStoriesPrefix.jsonl"

  if downloadData:
    count = downloadTinyStoriesPrefix(dataPath, size=10000)
    print(f"Saved {count} stories to {dataPath}")

  stories: list[str] = []
  if useToyData: stories = toyStories()
  else: stories = loadStoriesFromJsonl(dataPath, maxStories=maxStories)

  # 3. a.
  train, validation, test = splitStories(stories, seed=splitSeed)
  print(f"Stories - train: {len(train)}, validation: {len(validation)}, test: {len(test)}")
  # 3. a.

  # 3. b.
  trainSentences = sentenceSplitStories(train)
  validationSentences = sentenceSplitStories(validation)
  testSentences = sentenceSplitStories(test)

  merges = learnBPE(trainSentences, numMerges=numMerges)

  # Apply the training-set tokenizer to every split. Each item remains a
  # complete story, while every sentence receives explicit boundary tags.
  trainTokens = tokenizeStories(train, merges)
  validationTokens = tokenizeStories(validation, merges)
  testTokens = tokenizeStories(test, merges)

  tokenToID, idToToken = buildBPETokenMaps(trainSentences, merges)

  trainTokenCounts = collections.Counter(token for story in trainTokens for token in story)
  vocabSize = len(tokenToID)
  numTrainTokens = sum(trainTokenCounts.values())
  numValidationTokens = sum(len(story) for story in validationTokens)
  numTestTokens = sum(len(story) for story in testTokens)

  print(f"BPE merges learned: {len(merges)}")
  print(f"Vocabulary size: {vocabSize}")
  print(f"Train tokens: {numTrainTokens}")
  print(f"Validation tokens: {numValidationTokens}")
  print(f"Test tokens: {numTestTokens}")

  frequencies = sorted(trainTokenCounts.values(), reverse=True)
  ranks = range(1, len(frequencies) + 1)
  plt.figure()
  plt.loglog(ranks, frequencies, marker=".")
  plt.xlabel("Rank")
  plt.ylabel("Frequency")
  plt.title("Training Set Token Frequency vs. Rank")
  plt.tight_layout()
  plt.savefig("tokenFrequencyVsRank.png")
  plt.close()
  # 3. b.

  trainIDs = [encode(story, tokenToID) for story in trainTokens]

  # 3. c.
  example = "The quick brown fox jumped over the lazy dog"
  # the [0] takes the list of lists and extracts the single
  # list that we need for this case.
  examp_tokens = ["<s>", *tokenizeBPE([example], merges)[0], "</s>"]
  exampIDs = encode(examp_tokens, tokenToID)
  exampDecoded = decode(exampIDs, idToToken)

  print(f"Example tokenization: {examp_tokens}")
  print(f"Example IDs: {exampIDs}")
  print(f"Example decoded: {exampDecoded}")
  print(f"Example reconstructed: {''.join(exampDecoded[1:-1])}")
  # 3. c.

  X, Y = genBatch(trainIDs, B=2, T=10, rng=np.random.default_rng(batchSeed))

  decodedX = decode(X[0], idToToken)
  decodedY = decode(Y[0], idToToken)
  print(f"Batch X shape: {X.shape}")
  print(f"Batch Y shape: {Y.shape}")
  print(f"Example X IDs: {X[0].tolist()}")
  print(f"Example Y IDs: {Y[0].tolist()}")
  print(f"Example X decoded: {decodedX}")
  print(f"Example Y decoded: {decodedY}")

if __name__ == "__main__":
  main()
