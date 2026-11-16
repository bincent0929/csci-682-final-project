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

## Plan

**For next week:** plug `MorphemeTokenizer` into the SLM from a prior assignment.

**Future steps:**
- Use an LLM to segment words the database can't resolve, and save those results to the same dictionary file.
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
