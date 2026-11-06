# Related work and the novelty claim (N6)

Searched 2026-11-06. This note decides what the report may claim. Short version: **batching BPE merges is not new, and
neither is parallel BPE training.** What we did not find is a trainer that merges several pairs per parallel round and
proves that the merge list, including its order, is byte-identical to sequential BPE.

## What the report may say

> We found no prior BPE trainer that applies several merges per parallel round with a proof that the merge list,
> including order and tie-breaking, is byte-identical to sequential training. BatchBPE (Morgan, 2024) batches merges
> first, but changes the merge order and gives an informal safety argument rather than a proof.

Keep "we found" and give the search date. A search cannot show that something doesn't exist.

**The report must not say:**
- "first batched BPE" or "first multi-merge BPE": BatchBPE came first.
- "first parallel BPE training": HF `tokenizers` and YouTokenToMe both parallelise the merge step.
- "BatchBPE is wrong". We have not tested it. Its stated goal is the same tokenization, not the same merge list.
- that the independence idea is ours. Zouhar et al. characterise when merges can be reordered, and "run a prefix of
  non-interacting tasks each round" is the deterministic-reservations pattern from parallel algorithms.

**Our contribution, stated narrowly:**
1. The closed-form rule for how much of a tied prefix is safe, with the proof (`batched_merging.md`).
2. Exactness at the level of the merge list: order, tie-breaks and token ids, at every thread count.
3. A shared-memory implementation with hashed owners, measured against IncrSerial.
4. Layered verification: sequential oracle, an independent replay (`batchsim`), and mutation checks.

## Why the same order matters

BatchBPE's README says the merge order differs from serial training while the token set and tokenization stay the
same. So the same-tokenization claim doesn't settle ours. Order matters here because:
- A token's id comes from its merge position, so a reordered list gives different ids and a different `vocab.json`.
  Models trained on one list can't use the other.
- Stopping at a fixed vocabulary size cuts the list at a position, so order can decide which tokens survive. BatchBPE's
  own §3.6 identifies a final-batch difference of this kind.
- `merges.txt` can be compared by checksum against the baseline at every thread count. Exactness is this project's
  headline requirement, and that check is how it is enforced.

The `eax`/`df` counterexample shows the gap. Sequential BPE merges `(e,a)`, then `(ea,x)`, then `(d,f)`. A batch of
`(e,a)` and `(d,f)` gives a different order. Those two later merges don't conflict, so Zouhar et al.'s safe permutations
suggest the tokenization is unaffected, which fits BatchBPE's claim, while our rule (k=1 here) keeps the list identical.

## Prior work on BPE training

| Work | Merges per step | Merge step parallel? | Same output as sequential? |
|---|---|---|---|
| **ParaBPE `incr-batch`** | many, per closed-form rule | yes, whole rounds | proved; checked byte for byte |
| BatchBPE (Morgan, 2024) | "often 200 or more" | no: only data loading uses joblib | different order; same tokens and tokenization claimed; informal argument |
| HF `tokenizers` 0.23.2 | 1 | yes, words of one merge via `maybe_par_iter` | n/a: is the sequential algorithm |
| YouTokenToMe | 1, up to 2 in flight (pipelined, not batched) | yes, workers own contiguous word slices; capped at 8 threads | tie order depends on insertion unless `DETERMINISTIC_QUEUE` is set |
| SentencePiece | 1 | no: its thread pool only counts the initial bigrams | n/a |
| rustbpe 0.1.0 (nanochat) | 1 | no: counting and regex are parallel, applying merges isn't | n/a |
| Two-phase BoundlessBPE (2026) | not stated in the abstract | not stated | "identical results" to its own original, by a different route (aggregating supermerges) |

Notes:
- **IncrPar isn't new.** One merge at a time over owned words is what HF and YouTokenToMe do. YouTokenToMe's contiguous
  worker slices match the map whose 2.30 imbalance B5 measures, so the hashed-owner result may apply to it. We haven't
  tested that.
- **HF and rustbpe break ties by smaller pair id.** SPEC §4 has its own rule, so their merges can differ from ours on
  ties. That is a spec difference, not an error on either side.

## Theory
- **Zouhar et al., "A Formal Perspective on Byte-Pair Encoding" (2023).** Two valid merge sequences are equivalent if
  and only if a safe permutation maps one to the other, and it gives O(N log M) greedy BPE. This concerns what applying
  merges does. It has no multi-merge training rule and no tie-breaking rule. Appendix B.2 shows that taking the top-M
  pairs in one go, with no recounting, can be arbitrarily bad, which is why our rule never does that.
- **Blelloch, Fineman, Gibbons, Shun, "Internally deterministic parallel algorithms can be fast" (PPoPP 2012).**
  Deterministic reservations: each round handles a prefix of remaining tasks in reserve and commit phases. Our rounds
  have the same shape. The difference is that the BPE task list isn't known ahead of time, because a merge creates new
  pairs, and handling that is the point of `T` and the proof.
- **Blelloch, Fineman, Shun, "Greedy Sequential MIS and Matching are Parallel on Average" (SPAA 2012).** The analogy
  for span: greedy order is usually shallow in practice. Our 2,561 rounds against 31,743 merges is the same effect.

## Neighbouring problems
- **Re-Pair** is the same greedy idea applied to compression. Parallel Re-Pair exists (Matsushita and Inoguchi, DCC
  2022): threads run on blocks with a shared dictionary, reaching 7.9–10.4× on 32 processors. The page doesn't say
  whether the grammar matches sequential Re-Pair. Block-wise counting suggests it may not, but we haven't read the full
  paper.
- **GPU BPE** (BlockBPE, GPUTOK, LLBPE) is about encoding with a fixed merge table, not training.

## Limits of this search
- HF, SentencePiece, YouTokenToMe and rustbpe were read from source, but through a summarising fetch, so confirm any
  line quoted in the report. BatchBPE was read from its paper and README, not its code. **Check the §3.6 wording against
  the PDF before quoting it.**
- Two-phase BoundlessBPE was read at abstract level only. The full paper should be checked before S9, in case a step
  applies several merges at once.
- Matsushita and Inoguchi was read at abstract level only.
- HF was read at the 0.23.2 release on docs.rs, not the git head.
- English-language web search only. No patents, and nothing about closed-source trainers.

## Sources
- Morgan, *Batching BPE Tokenization Merges*: https://arxiv.org/abs/2408.04653; README: https://github.com/alexandermorgan/BatchBPE
- Zouhar et al., *A Formal Perspective on Byte-Pair Encoding*: https://arxiv.org/abs/2306.16837
- HF `tokenizers` 0.23.2 trainer: https://docs.rs/tokenizers/latest/src/tokenizers/models/bpe/trainer.rs.html
- YouTokenToMe: https://github.com/VKCOM/YouTokenToMe
- SentencePiece trainer: https://github.com/google/sentencepiece/blob/master/src/bpe_model_trainer.cc
- rustbpe 0.1.0: https://docs.rs/crate/rustbpe/latest/source/src/lib.rs
- *Faster Superword Tokenization* (two-phase BoundlessBPE): https://arxiv.org/abs/2604.05192
- Blelloch et al., PPoPP 2012: https://doi.org/10.1145/2145816.2145840
- Blelloch, Fineman, Shun, SPAA 2012: https://arxiv.org/abs/1202.3205
- Matsushita and Inoguchi, DCC 2022: https://sigport.org/documents/applying-practical-parallel-grammar-compression-large-scale-data
- BlockBPE: https://arxiv.org/abs/2507.11941; GPUTOK: https://arxiv.org/abs/2603.02597; LLBPE: https://arxiv.org/abs/2610.04951
