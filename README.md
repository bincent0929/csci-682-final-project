# CSCI 682 Final Project: Morpheme-Based Tokenization

Train a language model on **morphemes** instead of BPE tokens and compare the two.

## Motivation

- **BPE ignores meaning.** BPE splits words by frequency, so its pieces often cut across real morphemes (`un-happi-ness` vs. whatever the statistics produce). Whether linguistically informed splits help is an open research question ([arXiv:2410.12656](https://arxiv.org/html/2410.12656v4)).
- **People read from partial words.** Readers recognize a word from only some of its letters (typoglycemia; e.g. with vowels removed, "ths sntnc s stll rdbl" is still readable). We may be able to use this idea: a model could predict a whole word from some of its pieces.

## Approach

RNN / Transformer language model + morpheme splitting.

- Library: [`morphemes`](https://pypi.org/project/morphemes/#user-content-usage). It downloads a morpheme database and parses one word at a time with `.parse()`.
- See [src/final_project/morpheme_library_testing.py](src/final_project/morpheme_library_testing.py) for a minimal example. The database goes in `data/`, which git ignores.

### `MorphemeTokenizer` (Harsh's design, [src/final_project/morpheme_tokenizer.py](src/final_project/morpheme_tokenizer.py))

Calling `.parse()` on every word is too slow, so the tokenizer works like this:

1. **Build a word → morpheme dictionary once.** Load the library's `WORDS` table directly (`Morphemes(path).db.load_db("r").table("WORDS").all()`) and read each row's `MorphoLexSegm` segmentation.
2. **Align morphemes to the surface spelling.** Morphemes don't always match the spelling exactly, so the gaps between matched morphemes are kept as their own pieces. Joining the pieces always gives back the original word.
3. **Cache the dictionary** as `data/dictionary.json`, and load it on later runs instead of rebuilding it.
4. **Tokenize text** by splitting it into sentences (NLTK `sent_tokenize`) and wrapping each one in `<s>` … `</s>`. Each word is split into morphemes with its original casing kept, and punctuation becomes its own token.
5. **Words not in the database** are kept whole for now (`handleUndefined`). They are added to the dictionary and saved.

> **Note:** The next two sections ("Segmenting unknown words with an LLM" and "Using `MorphemeTokenizer` in the SLM") were written by an LLM to give direction for the implementation. They are suggestions, not decisions the team has made or code that exists yet, so check them before relying on them.

### Segmenting unknown words with an LLM

The database only covers words in MorphoLex, so names ("Lily"), made-up words, and many inflected forms fall through to `handleUndefined`, which keeps them whole. Each of those becomes its own vocabulary entry that the model sees only a few times. An LLM can split them instead.

1. **Collect unknown words in one pass, before training.** Run the tokenizer over the training corpus and record every word that isn't in the dictionary. Don't call the LLM from inside `splitWord`: it would be slow, and training would depend on a network call.
2. **Track unknown words separately.** Right now `handleUndefined` writes `[word]` into `dictionary.json`, so on the next run the word looks resolved and is never revisited. Keep unknown words in their own list (e.g. `data/unresolved.json`), or store LLM results in a separate `data/llm_dictionary.json`, so we can tell database splits, LLM splits, and fallbacks apart. That separation also makes it easy to compare the two sources later.
3. **Send words in batches** (e.g. 100 per request) and ask for JSON back. A prompt along these lines:
   ```text
   Split each English word into its morphemes (prefixes, roots, suffixes).
   The pieces must join back into the exact word, letter for letter.
   Names and words with no internal structure stay as one piece.
   Return only JSON: {"unhappiness": ["un", "happi", "ness"], "lily": ["lily"]}

   Words: unhappiness, lily, jumpy, ...
   ```
   Giving a few examples taken from the database (`dictionary.json`) helps the LLM match MorphoLex's style of segmentation.
4. **Validate every answer** before saving it: the pieces must be non-empty and `"".join(pieces) == word`. This is the same rule `resolveMorphemes` follows. If a split fails the check, retry it once, then keep the word whole.
5. **Save the results and reuse them.** Merge the validated splits into the dictionary that `MorphemeTokenizer` loads, so each word is only sent to the LLM once. Because the splits are saved, every later run is reproducible and offline.
6. **Fallback for words seen only at test time:** keep the word whole (the model will map it to `<unk>`), or split it into single characters so the model can still read it. Characters are the safer choice because the vocabulary stays closed.

A cheap model is fine for this. TinyStories uses a small vocabulary, so the list of unknown words should only be a few thousand.

### Using `MorphemeTokenizer` in the SLM

The SLM in [src/final_project/assignment3_slm_files/](src/final_project/assignment3_slm_files/) already has the right shape for this. `MorphemeTokenizer.tokenize(stories)` returns one token list per story, with `<s>` … `</s>` around each sentence. That is the same format as `tokenizeStories(stories, merges)`, the BPE function `prepData` calls in [slm.py](src/final_project/assignment3_slm_files/slm.py). The transformer itself doesn't change: it only sees integer IDs and a `vocab_size`.

1. **Swap the tokenizer in `prepData`.** Replace `learnBPE` and the two `tokenizeStories` calls:
   ```python
   from final_project.morpheme_tokenizer import MorphemeTokenizer

   tokenizer = MorphemeTokenizer("data")  # repo-root data/ folder; builds dictionary.json on the first run
   trainStories_tokens = tokenizer.tokenize(trainStories)
   validStories_tokens = tokenizer.tokenize(validationStories)
   ```
   Everything after that stays the same: `buildTokenMaps`, `encode`, `genBatch`, `train`, and `TransformerLM(vocab_size=len(tokenToID), ...)`.
2. **Add a `tokenizer` switch** (`"bpe"` or `"morpheme"`) to `prepData`/`main` rather than deleting the BPE path. Both models can then be trained with the same settings, which is what the comparison needs.
3. **Fix the output formatting.** `formatSample` removes the BPE `</w>` marker. Morpheme tokens keep spaces and punctuation as their own tokens, so `"".join(tokens)` already rebuilds the text. Only `<s>`/`</s>` need handling. A `formatMorphemeSample` (or a branch in `formatSample`) covers this.
4. **Expect a different vocabulary and sequence length.**
   - The vocabulary comes from every distinct morpheme, space, punctuation mark, and whole unknown word in the training set, not from a fixed merge count. Print `len(tokenToID)` for both tokenizers.
   - Casing is kept, so `The` and `the` are different tokens. Lowercasing everything, or adding a `<cap>` marker token before capitalized words, would shrink the vocabulary.
   - Spaces are tokens, so a 32-token context covers fewer words than it does with BPE. Either raise `contextSize` or attach the space to the next word's first morpheme (`" un"`, `"happi"`, `"ness"`) to cut the sequence length roughly in half.
   - Validation morphemes that never appear in training map to `<unk>`. The LLM step above and the character fallback reduce how often that happens.
5. **Compare fairly.** Raw cross-entropy per token isn't comparable across tokenizers, because the tokens differ in size. Report **bits per character**: total validation loss (in nats) ÷ number of characters in the validation text ÷ ln 2. Use the same stories, split seed, model size, and training steps for both runs.
   - **Add a morpheme-heavy test set.** Besides the random TinyStories split, pick or build test data where a large share of words are found in the morpheme dictionary. Good candidates are words with clear prefixes and suffixes (`un-`, `re-`, `-ness`, `-ful`, `-ly`). A simple filter: keep the stories or sentences where most words are in `dictionary.json` and split into two or more morphemes. The best case is words whose morphemes appear in training but whose full word does not (training has `kind` and `-ness`, the test set has `kindness`). This is where the morpheme model should do best, because BPE has no reason to split such a word at its real boundaries.
   - Report results on **both** test sets. The random split shows how the model does on ordinary text. The morpheme-heavy set shows the case where morphemes should help. On its own, a test set picked to favor morphemes would overstate the benefit, so it shouldn't be the only result.
6. **Check the data path.** `slm.py` reads `Path(__file__).resolve().parents[2] / "data"`, which resolves to `src/data/`, not the repo-root `data/` folder. Use `parents[3]` so the stories and the morpheme dictionary live in the same git-ignored `data/` folder.

## Plan

**For next week:** plug `MorphemeTokenizer` into the SLM from a prior assignment (see [Using `MorphemeTokenizer` in the SLM](#using-morphemetokenizer-in-the-slm)).

**Future steps:**
- Use an LLM to segment words the database can't resolve (see [Segmenting unknown words with an LLM](#segmenting-unknown-words-with-an-llm)).
- Compare a morpheme-based model with a token-based (BPE) model.
- Try architecture tweaks.

## Setup

The project uses **[uv](https://docs.astral.sh/uv/)**, a fast Python package and project manager (it replaces `pip` and `venv`). It reads the dependencies from `pyproject.toml`, installs the exact versions pinned in `uv.lock` into a local `.venv/`, and runs commands inside it. That way everyone has the same packages.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. From the repo root, run:
   ```sh
   uv sync                     # installs Python 3.14 (from .python-version) and all dependencies
   source .venv/bin/activate   # optional: otherwise prefix commands with `uv run`
   ```

### After setup

- One-time NLTK data download, used by `sent_tokenize`:
  ```sh
  python -c "import nltk; nltk.download('punkt_tab')"
  ```
- Run a script: `uv run python src/final_project/morpheme_library_testing.py`. The first run downloads the morpheme database into `data/`.
- Add a dependency: `uv add <package>`. This updates both `pyproject.toml` and `uv.lock`, so commit both files.
