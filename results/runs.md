### Morphizer Model
```text
PS C:\Users\harsh\Code\682\Final\src\final_project> uv run TrainLM.py
training stories: 160000
validation stories: 20000
test stories: 20000
device: cuda
vocabulary size: 13600
encoded training stories: 160000
encoded validation stories: 20000
step 1: train loss 9.7067, validation loss 7.5475
step 1000: train loss 1.7042, validation loss 1.7587
step 2000: train loss 1.5440, validation loss 1.5783
step 3000: train loss 1.4213, validation loss 1.4543
step 4000: train loss 1.4519, validation loss 1.3713
step 5000: train loss 1.3614, validation loss 1.4236
step 6000: train loss 1.2834, validation loss 1.3432
step 7000: train loss 1.3043, validation loss 1.3231
step 8000: train loss 1.2740, validation loss 1.2642
step 9000: train loss 1.3140, validation loss 1.3061
step 10000: train loss 1.1148, validation loss 1.2945
step 11000: train loss 1.2408, validation loss 1.2240
step 12000: train loss 1.2825, validation loss 1.2403
step 13000: train loss 1.2661, validation loss 1.1852
step 14000: train loss 1.2251, validation loss 1.2226
step 15000: train loss 1.1586, validation loss 1.1997
step 16000: train loss 1.2614, validation loss 1.2099
step 17000: train loss 1.1635, validation loss 1.1722
step 18000: train loss 1.1813, validation loss 1.1832
step 19000: train loss 1.1182, validation loss 1.1818
step 20000: train loss 1.1615, validation loss 1.1749
sample 1:
It could not help them.
It wanted to be free and good for your feelings.
But it was too late.
After
sample 2:
But then, a big dog came runing towards them and fell down.
The dog ran away to get his vest.
Lily
sample 3:
She tryed to open it, but the menu was too heavy.
She asked her mom, "Can I weigh it to improve?"
sample 4:
For the book, the magazine.
She saw a page that said that she had to help people who could help too.
But this
sample 5:
The voice tryed to stop, but it didn't stop.
The bird kept runing until it was at a big tree.
It flew
```

### BPE Model
```text
PS C:\Users\harsh\Code\682\Final\src\final_project> uv run TrainBPE.py
training stories: 160000
validation stories: 20000
test stories: 20000
device: cuda
vocabulary size: 321
encoded training stories: 160000
encoded validation stories: 20000
step 1: train loss 5.8720, validation loss 5.3549
step 1000: train loss 1.8540, validation loss 1.8330
step 2000: train loss 1.6502, validation loss 1.6470
step 3000: train loss 1.4262, validation loss 1.5534
step 4000: train loss 1.4868, validation loss 1.4745
step 5000: train loss 1.3016, validation loss 1.4646
step 6000: train loss 1.4476, validation loss 1.3877
step 7000: train loss 1.4452, validation loss 1.3857
step 8000: train loss 1.3739, validation loss 1.3770
step 9000: train loss 1.3769, validation loss 1.3181
step 10000: train loss 1.3760, validation loss 1.3295
step 11000: train loss 1.3308, validation loss 1.3196
step 12000: train loss 1.3301, validation loss 1.3144
step 13000: train loss 1.2903, validation loss 1.3167
step 14000: train loss 1.2875, validation loss 1.2972
step 15000: train loss 1.2702, validation loss 1.2807
step 16000: train loss 1.3612, validation loss 1.2825
step 17000: train loss 1.2717, validation loss 1.2696
step 18000: train loss 1.3954, validation loss 1.2727
step 19000: train loss 1.2829, validation loss 1.2891
step 20000: train loss 1.2600, validation loss 1.2563
sample 1:
Once upon a time there was a little girl named Lily.
She loved watching around in the garden and playing.
One day, she went to the park to
sample 2:
Bobby was so excited that he asked the rock if he could run too.
The rock together, but they couldn't la
sample 3:
It shook his head.
"No, Ben, no!
This is my picture.
It's too hard for your
sample 4:
And you have to share and be patient.
Do you understand?"
The teacher agreed and said, "Yes, I will
sample 5:
He was so surprised.
He dodged very ignorant and went to eat his crystal crystal.
He was
```