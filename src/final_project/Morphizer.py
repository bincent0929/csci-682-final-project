import json
import re
from pathlib import Path
import ollama
from nltk.tokenize import sent_tokenize

class Morphizer:
  def __init__(self, model="qwen2.5:3b"):
    self.dataPath = Path(__file__).resolve().parents[1] / "data"
    self.dataPath.mkdir(parents=True, exist_ok=True)
    self.encoderPath = self.dataPath / "dictionary.json"
    self.encoder = {}
    self.decoder = {}

    if self.encoderPath.exists(): self.encoder = json.loads(self.encoderPath.read_text())
    else: self.buildEncoder()
    self.buildDecoder()
    
    self.morphs = set()
    for pieces in self.encoder.values():
      if len(pieces) > 1:
        for piece in pieces:
          self.morphs.add(piece)
    self.model = model
    
  def buildEncoder(self):
    for line in (self.dataPath / "morphemepiece.tsv").read_text(encoding="utf-8").splitlines():
      word, morphs = line.split("\t")
      self.encoder[word] = morphs.split(" ")
    self.save()
  def buildDecoder(self):
    for a, b in self.encoder.items():
      #if a == "happyness":
      #  print(a, b)
      self.decoder[tuple(b)] = a

  def handleUndefined(self, word):
    prompt = ("Split each English word into morphemes with +.\n"
              "unhappiness -> un+happy+ness\n"
              "rebuilding -> re+build+ing\n"
              "disagreement -> dis+agree+ment\n"
              "nationality -> nation+al+ity\n"
              "unbelievable -> un+believe+able\n"
              "rabbit -> rabbit\n"
              "coded -> code+ed\n"
              "lily -> lily\n"
              "sunflower -> sun+flower\n"
              "garden -> garden\n"
              + word + " ->")

    response = ollama.generate(model=self.model, prompt=prompt, raw=True,
                               options={"temperature": 0, "num_predict": 20, "stop": ["\n"]})
    pieces = []
    for piece in response["response"].split("+"):
      pieces.append(piece.strip())

    # Ensuring that all of our pieces combing to form our original word (LLM output validation)
    if "".join(pieces) != word:
      return [word]

    # We see almost all sub-3 char morphemes in our database so if we end up with a new one here then we clearly have an issue in our LLM output.
    for piece in pieces:
      if piece not in self.morphs and len(piece) < 3:
        return [word]
    return pieces

  def save(self):
    self.encoderPath.write_text(json.dumps(self.encoder))

  def splitWord(self, word):
    if word.lower() not in self.encoder: self.encoder[word.lower()] = self.handleUndefined(word.lower())
    if "".join(self.encoder[word.lower()]) != word.lower():
      pieces = list(self.encoder[word.lower()])
      if word[0].isupper(): pieces[0] = pieces[0].capitalize()
      return pieces
    pieces = []
    start = 0
    for piece in self.encoder[word.lower()]:
      pieces.append(word[start:start + len(piece)])
      start += len(piece)
    return pieces
  def recombineWord(self, morphs):
    key = tuple(morphs)
    if key in self.decoder:
      return self.decoder[key]
    else:
      return "".join(morphs)

  def tokenize(self, texts):
    tokenized = []
    for text in texts:
      tokens = []
      for sent in sent_tokenize(text):
        tokens.append('<s>')
        for word, remaining in re.findall(r"([A-Za-z]+(?:'[A-Za-z]+)?)|(.)", sent, flags=re.DOTALL):
          if word: tokens += self.splitWord(word)
          else: tokens.append(remaining)
        tokens.append('</s>')
      tokenized.append(tokens)
    self.save()
    return tokenized
  