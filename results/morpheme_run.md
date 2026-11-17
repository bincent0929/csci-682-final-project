# Morpheme Tokenizer Run

First full run of the SLM with `MorphemeTokenizer` replacing BPE on TinyStories (November 16, 2026).

## Setup

- **Code:** `main` at `78875cf`, plus the `slm.py` change that replaces BPE with `MorphemeTokenizer` in `prepData`. The BPE version is the previous `slm.py`.
- **Command:** `uv run python -m final_project.assignment3_slm_files.slm`
- **Machine:** Windows, CPU (the torch CPU build installed by uv), Python 3.14.
- **Data:** the first 1,000 stories of `src/data/tinyStories.jsonl` (downloaded with `downloadTinyStoriesPrefix`), randomly split into 800 training, 100 validation and 100 test stories.
- **Model:** `TransformerLM` with `n_embed=64`, `n_head=4`, `n_layer=2` and a context size of 32.
- **Training:** 1,000 steps of AdamW, batch size 16, learning rate 1e-3.
- **Evaluation:** the training loss is from the single batch at that step. The validation loss is averaged over 5 random batches every 100 steps.

## Results

| Metric | Value |
|---|---|
| Vocabulary size | 3,098 |
| Final validation loss (step 1000) | 2.4073 |

| Step | Training loss | Validation loss |
|---|---|---|
| 1 | 8.2905 | 8.0858 |
| 100 | 3.3502 | 3.1763 |
| 200 | 2.8580 | 3.0317 |
| 300 | 2.9222 | 2.8885 |
| 400 | 2.7843 | 2.8638 |
| 500 | 2.7201 | 2.7115 |
| 600 | 2.6841 | 2.6675 |
| 700 | 2.5970 | 2.6705 |
| 800 | 2.6189 | 2.5008 |
| 900 | 2.4055 | 2.4950 |
| 1000 | 2.4389 | 2.4073 |

![Training and validation loss](slm_loss_morpheme.png)

## Generated Samples

Each sample starts from `<s>` and generates 50 tokens. Line breaks are sentence ends.

```text
sample 1:
Jill wanted the journey momes.
She want to it to with break play any.
She saw the l startn like."
sample 2:
Heryside you fell the new park for him.
They will were loud so !"
Ben laughd and held welcome, there are
sample 3:
He look.
They noise was not Gotcha!"
The Maryped you dough head the adventure to play name're him."
She said
sample 4:
He mom for see.
His were old card for not feel inusing and pocket to tell all the ball.
She said, "
sample 5:
She feed with luggages lunch."
She the little mom!
"Let's be safe fire dinner?""
The tears, there said a
```

## Notes

- **The starting loss is what it should be.** The step 1 validation loss (8.09) is close to ln(3098) = 8.04, the loss of guessing uniformly over the vocabulary, which is what an untrained model should give.
- **No sign of overfitting.** The validation loss stays close to the training loss for all 1,000 steps.
- **Some generated words join a root with the wrong ending**, such as "laughd", "momes" and "luggages". "laughd" likely comes from how the tokenizer handles roots ending in "e": "scared" is split into "scare" + "d", while "laughed" is split into "laugh" + "ed", so the past tense has two different tokens, and the model attached "d" to a root that takes "ed".
- **Results vary between runs.** The split has no seed, so each run trains and validates on different stories, and the numbers can't be reproduced exactly.
- **No baseline yet.** The per-token losses here can't be compared with the BPE version, because the same text becomes a different number of tokens. A fair comparison needs the same split in both versions and a per-character measure such as bits per character.
- **Not measured yet:** how many validation tokens became `<unk>`. Words that only appear in validation become `<unk>`, which is easy to predict, so this can make the morpheme model look better than it is.
