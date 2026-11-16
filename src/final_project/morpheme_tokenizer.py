import json
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize
from morphemes import Morphemes

class MorphemeTokenizer:
  def __init__(self, dataPath):
    Path(dataPath).mkdir(parents=True, exist_ok=True)
    self.dictionaryPath = Path(dataPath) / "dictionary.json"
    self.dictionary = {}
    if self.dictionaryPath.exists(): self.dictionary = json.loads(self.dictionaryPath.read_text())
    else: self.buildDictionary(dataPath)

  def buildDictionary(self, dataPath):
    rows = Morphemes(dataPath).db.load_db("r").table("WORDS").all()
    for row in rows:
      word = str(row["Word"]).lower()
      morphs = re.findall(r"[a-z']+", row["MorphoLexSegm"])
      self.dictionary[word] = self.resolveMorphemes(word, morphs)
    self.save()

  def resolveMorphemes(self, word, morphs):
    pieces = []
    for morph in morphs:
      i = word.find(morph)
      if i < 0: continue
      if i > 0: pieces.append(word[:i])
      pieces.append(morph)
      word = word[i + len(morph):]
    if word: pieces.append(word)
    return pieces

  def handleUndefined(self, word):
    return [word]

  def save(self):
    self.dictionaryPath.write_text(json.dumps(self.dictionary))

  def splitWord(self, word):
    if word.lower() not in self.dictionary: self.dictionary[word.lower()] = self.handleUndefined(word.lower())
    pieces = []
    start = 0
    for piece in self.dictionary[word.lower()]:
      pieces.append(word[start:start + len(piece)])
      start += len(piece)
    return pieces

  def tokenize(self, stories):
    C = []
    for story in stories:
      tokens = []
      for sent in sent_tokenize(story):
        tokens.append('<s>')
        for word, other in re.findall(r"([A-Za-z]+(?:'[A-Za-z]+)?)|(.)", sent, flags=re.S):
          if word: tokens += self.splitWord(word)
          else: tokens.append(other)
        tokens.append('</s>')
      C.append(tokens)
    self.save()
    return C
