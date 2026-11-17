# CSCI 682 Final Project: Morpheme-Based Tokenization

Train a language model on **morphemes** instead of BPE tokens and compare the two.

## Running the Program

The project uses **[uv](https://docs.astral.sh/uv/)**, a fast Python package and project manager (it replaces `pip` and `venv`). It reads the dependencies from `pyproject.toml`, installs the exact versions pinned in `uv.lock` into a local `.venv/`, and runs commands inside it. That way everyone has the same packages.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. From the repo root, run:
   ```sh
   uv sync                     # installs Python 3.14 (from .python-version) and all dependencies
   source .venv/bin/activate   # optional: otherwise prefix commands with `uv run`
   ```

### Downloading data and running scripts

- One-time NLTK data download, used by `sent_tokenize`:
  ```sh
  python -c "import nltk; nltk.download('punkt_tab')"
  ```
- Run a script: `uv run python src/final_project/morpheme_library_testing.py`. The first run downloads the morpheme database into `data/`.
- Download TinyStories for the SLM (the command may print an error, but check `src/data/`, because the file is often downloaded anyway):
  ```sh
  mkdir -p src/data
  uv run python -c "from final_project.assignment3_slm_files.tinyStoriesPipeline import downloadTinyStoriesPrefix; print(downloadTinyStoriesPrefix('src/data/tinyStories.jsonl', size=10000))"
  ```
- Train the SLM with the morpheme tokenizer: `uv run python src/final_project/assignment3_slm_files/slm.py`. The first run builds `data/dictionary.json`. The loss plot is saved as `slm_loss.png` in the folder you run it from.
  > **Note:** The script runs on the **CPU**, even if you have a GPU, so that every run gives exactly the same results (see [Reproducible runs](#reproducible-runs)). A run takes about 10 seconds. To train on a GPU (CUDA or MPS) instead, set `deterministic = False` in `main` in `slm.py`. The results will then vary slightly from run to run.
- Add a dependency: `uv add <package>`. This updates both `pyproject.toml` and `uv.lock`, so commit both files.

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

### `MorphemeTokenizer` in the SLM (Harsh, commit `571a2a2`)

`prepData` in [slm.py](src/final_project/assignment3_slm_files/slm.py) now tokenizes with `MorphemeTokenizer` instead of BPE:

```python
morphemeTokenizer = MorphemeTokenizer(str(Path(__file__).resolve().parents[3] / "data"))
trainStories_tokens = morphemeTokenizer.tokenize(trainStories)
validStories_tokens = morphemeTokenizer.tokenize(validationStories)
```

The rest of the pipeline is unchanged: `buildTokenMaps`, `encode`, `genBatch`, `train`, and `TransformerLM`. The existing `formatSample` already prints morpheme output correctly, because it only joins the tokens, turns `</s>` into a newline, and removes `</w>`, which never appears in morpheme output.

Current state of the code:
- **The BPE path was replaced, not kept behind a switch.** `learnBPE` and `tokenizeStories` are still imported, but nothing calls them. The `numMerges` and `bpeTrainStories` arguments to `prepData` are now unused. To get a BPE baseline, we have to go back to the `slm.py` from before `571a2a2`.
- **The data comes from two different folders.** The morpheme dictionary is read from the repo-root `data/` folder (`parents[3]`). `main` still reads the stories from `src/data/tinyStories.jsonl` (`parents[2]`).

### Reproducible runs

All random choices in `slm.py` use one seed (`seed = 123` in `main`): the train/validation/test split, the model's starting weights, the training batches, the validation batches, and the sampling of generated text. Every validation estimate uses the same batches, so the validation losses at different steps are measured on the same text.

By default the SLM runs on the CPU with `torch.use_deterministic_algorithms(True)` (`deterministic = True` in `main`). Two runs give exactly the same output. The GPU (CUDA or MPS) isn't reproducible even with the seed: on MPS the losses started to differ by about 0.001 after step 200, because the GPU doesn't always add numbers in the same order. For this small model, the CPU is also slightly faster (about 10 s per run). Set `deterministic = False` to train on a GPU.

### Morpheme run results

See [results/morpheme_run.md](results/morpheme_run.md) for the full record and [the loss plot](results/slm_loss_morpheme.png). The run used 1,000 TinyStories split 800/100/100, a 2-layer model with 64-dimensional embeddings, 4 heads, a context size of 32, and 1,000 training steps.

| Metric | Value |
|---|---|
| Vocabulary size | 3,146 |
| Validation loss at step 1 | 8.08 (ln 3146 = 8.05) |
| Validation loss at step 1000 | 2.46 |

At step 1, the validation loss is close to ln(V), the natural log of the vocabulary size, which is the loss of a uniform guess. That's what an untrained model should give. See [docs/starting_loss.md](docs/starting_loss.md) for why.

What the run showed:
- **No overfitting.** The validation loss stays close to the training loss for all 1,000 steps.
- **Some generated words attach the wrong ending to a root**, such as "scareed", "curiouss" and "relieveful". This likely comes from how roots ending in "e" are split: "scared" becomes `scare` + `d`, while "laughed" becomes `laugh` + `ed`. The past tense ends up as two different tokens, so the model can attach an ending to a root that needs a different one.
- **The run can't be compared with BPE yet.** Per-token loss depends on the tokenizer. A fair comparison needs the same split for both tokenizers and a per-character measure such as bits per character (see step 5, "Compare fairly", in [docs/llm_suggestions.md](docs/llm_suggestions.md#using-morphemetokenizer-in-the-slm)).
- **The number of `<unk>` tokens wasn't measured.** Words that appear only in validation become `<unk>`, which is easy to predict, so the morpheme model's loss may look better than it really is.

### Suggested next steps (LLM-written)

[docs/llm_suggestions.md](docs/llm_suggestions.md) has two LLM-written guides: how to split unknown words with an LLM, and how to use `MorphemeTokenizer` in the SLM, including how to compare it fairly with BPE. They are suggestions, not decisions the team has made.

## Plan

**Done:**
- Plugged `MorphemeTokenizer` into the SLM from a prior assignment and recorded a run (see [Morpheme run results](#morpheme-run-results)).
- Seeded every random choice and made runs reproducible on the CPU (see [Reproducible runs](#reproducible-runs)).

**Next:**
- Bring back BPE behind a `tokenizer` switch and report bits per character, so the two tokenizers can be compared with the same seed.
- Count how many validation tokens become `<unk>`.
- Look at how the tokenizer splits roots ending in "e" (`scare` + `d` vs. `laugh` + `ed`).

**Future steps:**
- Use an LLM to segment words the database can't resolve (see [Segmenting unknown words with an LLM](docs/llm_suggestions.md#segmenting-unknown-words-with-an-llm)).
- Try architecture tweaks.
