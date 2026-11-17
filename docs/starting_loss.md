# Why the Starting Loss Is ln(V)

At step 1, the validation loss should be close to ln(V), the natural log of the vocabulary size V. That is the loss of a uniform guess, which is what an untrained model gives. The numbers below come from our [morpheme run](../results/morpheme_run.md), which had V = 3,146.

- **The loss** at each position is the cross-entropy loss −ln(p), where p is the probability the model gave the correct next token. The reported loss is the average over all positions.
- **An untrained model guesses uniformly.** Its starting weights are small and random, so its output scores (logits) are all close to 0, and softmax turns equal scores into equal probabilities: every token gets p = 1/V, where V is the vocabulary size.
- **So the starting loss is ln(V):** loss = −ln(1/V) = ln(V). Here, ln(3146) ≈ 8.05, and the measured loss was 8.08. It is slightly higher because the logits aren't exactly 0, and any uneven guess made without information does a little worse than a uniform one.
- **Why it's a useful check:** a starting loss well above ln(V) points to a setup problem, such as weights initialized too large. A starting loss well below ln(V) usually means the targets are leaking into the inputs.
- **Perplexity** is e^loss. At step 1 it is e^ln(V) = V, so the model is as unsure as if it were picking from all 3,146 tokens at random. At step 1000, e^2.46 ≈ 11.7, so it is about as unsure as if it were picking from 12 equally likely tokens.
