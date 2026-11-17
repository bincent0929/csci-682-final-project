# Morpheme Tokenizer Run

```sh
uv run python src/final_project/assignment3_slm_files/slm.py
```

Use this to locally download 1000 tinyStories for the testing:
```sh
mkdir -p src/data
uv run python -c "from final_project.assignment3_slm_files.tinyStoriesPipeline import downloadTinyStoriesPrefix; print(downloadTinyStoriesPrefix('src/data/tinyStories.jsonl', size=10000))"
```

You may get an error, but check in the data folder, it may have downloaded anyway.

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