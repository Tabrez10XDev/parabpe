# Batched merging: exact parallel merge rounds

Serial BPE makes about 32k dependent merges, one per step, and that chain is the Amdahl term of phase 3: IncrPar reaches
only 1.14× at 5 threads. This note describes `incr-batch`. Each round merges several pairs at once, in one parallel
step, and the theorem below proves that the output is byte-identical to sequential merging. SPEC.md does not change:
batching is an execution strategy, not a different training rule.

## The rule

Sort the live pairs best-first (count descending, then the SPEC §4 tuple descending): q0, q1, q2, … All indices in
this note are **0-based**.

1. **Prefix.** p[0..K) is the longest prefix of that order whose pairs are pairwise token-disjoint, with K no larger
   than the remaining merge budget. A self-pair (a,a) may only be the last member, so it ends the prefix.
2. **Ties.** Let c* = count(p[K−1]). Let T be every live pair ranked after p[K−1] with count exactly c*. **T starts at
   the entry that stopped the prefix**, if that entry has count c*.
3. **Keep.** Let g be the number of members with count > c*. Let j be the first member index whose tokens appear in a
   pair of T, or K if there is none. Then

       k = max(min(K, j + 1), g, 1)

   If the scan of T hit its cap before T ended, use k = max(g, 1). That is exact for any cap, including 0.

Merge p[0..k) in one step; p[i] gets the id z_first + i.

**Mind the index.** With 1-based members p1..pK, the same formula is k = max(min(K, j), g, 1), where j is now the
1-based index. Mixing the two (1-based p, but j+1) keeps one member too many and is wrong. See the counterexample below.

## Theorem

**Starting from any training state S, the next k merges of sequential BPE are exactly p[0], …, p[k−1], in that order.**

Notation: ≻ is the SPEC §4 order; count_S is the count in S; p[i] = (a_i, b_i) → z_i; S_i is S after merging
p[0..i), so S_0 = S.

**Assumption.** ≻ is a strict total order on live pairs, meaning no two distinct pairs tie on count and on the byte
tuple. Two distinct token ids with identical bytes would break this, but SPEC §4 is then ambiguous for *every* engine,
and `format_vocab` throws on the duplicate key, so such a run fails loudly instead of producing a wrong file.

**Lemma 1 (old pairs never rise).** Merging (a,b) → z with z fresh leaves count(c,d) ≤ its old value for every pair of
old tokens, and sets count(a,b) to 0.

*Proof.* The rewrite only replaces adjacent `a b` by `z`, so two old tokens that are adjacent afterwards were adjacent
before, at a position determined by the new one. Greedy left-to-right rewriting leaves no `a b` behind: for a ≠ b every
occurrence is rewritten; for a self-pair (a,a), a run of length n becomes ⌊n/2⌋ z's plus at most one a, with no adjacent
a's left. ∎

**Lemma 2 (disjoint merges don't interact).** If {c,d} ∩ {a,b} = ∅, merging (a,b) leaves count(c,d) unchanged.

*Proof.* An occurrence of `c d` shares no position with an occurrence of `a b`, so the rewrite neither removes it nor
creates one. ∎

**Lemma 3 (witness for new pairs).** Let i ≤ K−1. Let r = (u,v) be a pair of S_i with at least one new token. Then there
is a pair O of S-tokens such that

- count_{S_i}(r) ≤ count_S(O),
- O shares a token with some member p[l] with l < i,
- O is not a member of p[0..K).

*Proof.* Members are pairs of S-tokens and are disjoint, so the new tokens never nest: each z_l expands to exactly
`a_l b_l` in S. Define s = b_l if u = z_l, otherwise s = u. Define t = a_m if v = z_m, otherwise t = v. Every occurrence
of (u,v) in S_i expands to an occurrence of O = (s,t) in S, at a position it determines, which gives the count bound.
O contains b_l or a_m with l, m < i, which gives the shared token.

Suppose O were a member p[n]. It shares a token with p[l] (or p[m]), so disjointness forces n = l (or n = m).

- u = z_l, v old: O = (b_l, v) = (a_l, b_l), so a_l = b_l.
- v = z_m, u old: O = (u, a_m) = (a_m, b_m), so a_m = b_m.
- u = z_l, v = z_m: O = (b_l, a_m) is p[l] and p[m], so l = m and b_l = a_l.

Each case makes a member of index < i ≤ K−1 a self-pair, but only p[K−1] may be one. ∎

These are the bounds (x,z) ≤ (x,a), (z,y) ≤ (b,y), (z_i,z_j) ≤ (b_i,a_j) and (z,z) ≤ (b,a) in one statement.

**Proof of the theorem.** By induction on i < k: p[i] is the sequential argmax in S_i. For i = 0, p[0] = q0. For i > 0,
Lemma 2 gives count_{S_i}(p[i]) = count_S(p[i]), since p[i] is disjoint from p[0..i). Take any other live pair r of S_i.

*Case 1: r has only S-tokens.* By Lemma 1 its count did not rise, and r is not one of p[0..i), whose counts are 0. So
r ranked after p[i] in S. Its count can only have dropped and its bytes are unchanged, so r still ranks after p[i].

*Case 2: r contains a new token.* By Lemma 3 there is a witness O outside the prefix, so O ranks after p[K−1]. So
count_S(O) ≤ c*, and equality would put O in T.

- If i < g: count(p[i]) > c* ≥ count_S(O) ≥ count(r), so p[i] wins on count.
- If i ≥ g: i < k forces k > g and k > 1 (here i > 0), so k = min(K, j+1) and i ≤ j. So p[0..i) all have index < j, and
  none of their tokens appears in T. O shares a token with one of them, so O ∉ T, so count_S(O) < c* = count(p[i]).

Either way p[i] beats r. ∎

**Exact for any cap.** An incomplete scan gives k = max(g, 1). Then every i with 0 < i < k has i < g, which never
needs T.

**Budget.** A shorter budget is only a shorter prefix; the proof does not change.

**Special tokens.** Their ids never occur in words, so they never appear in a pair.

**One pass per word.** Applying p[0], …, p[k−1] in turn equals one left-to-right pass that rewrites any member at the
current position. Members are disjoint, so occurrences of different members never overlap, and each token starts at most
one member. Only a self-pair can overlap itself, and the pass resolves that run left to right exactly as the greedy rule
does. So a round needs one whole-word diff per word.

**Zero-count pairs stay zero.** After a pair's count reaches 0, Lemma 1 keeps it at 0: its tokens already exist, and
rewriting never creates a new adjacency of old tokens. This makes the deferred index sweep in B4 safe.

## Why each condition is there

**Why the comparison with c* must be strict.** On a tie, (x,z) *beats* its witness (x,a), because bytes(z) extends
bytes(a). A witness of equal count therefore bounds nothing.

**Counterexample for keeping one member too many.** Take the words `eax`×2 and `df`×2 (separate documents).

| | |
|---|---|
| Live pairs, all count 2 | (e,a) ≻ (d,f) ≻ (a,x) |
| Prefix | p[0] = (e,a), p[1] = (d,f), and (a,x) stops it: K = 2, c* = 2, g = 0 |
| T | {(a,x)}; it shares `a` with p[0], so j = 0 |
| Correct k | max(min(2, 1), 0, 1) = 1 |

After (e,a) → `ea`, sequential BPE merges (ea,x), because ("ea","x") ≻ ("d","f"). The batch must stop at one member.
Note that T had to include (a,x) even though it was the entry that stopped the prefix.

**Why a self-pair must be last.** A self-pair is its own witness, so it isn't bounded by anything outside the batch. With
`aaa`×100 and `bc`×50: p[0] = (a,a) has count 200 and p[1] = (b,c) has count 50. After merging (a,a), the pair (aa,a) has
count 100 and beats (b,c). As the last member, its new pairs only compete from the next round on.

**Why p[j]'s own tokens need no check.** Nothing in the round comes after p[j] when k = j+1. This is the gain over the
relaxed rule, and it is large (see below).

## The three rules

Built in this order, each verified before the next:

| rule | k | note |
|---|---|---|
| strict | K if T is empty, else max(g,1) | no tie at c* at all |
| relaxed | max(min(K,j), g, 1) | no T pair touches any kept member |
| closed form | max(min(K,j+1), g, 1) | the last kept member's tokens are not checked; **ships** |

## Measurements (B0, `tools/batchsim`)

`batchsim` replays exact BPE from `words.tsv` with a `std::set` order and no project code. Each round computes a batch,
then applies it one merge at a time and asserts that every member is the sequential argmax. Its merges are byte-identical
to IncrSerial's `merges.txt` on `wikitext.10M` v4000. Every run below had 0 violations; round counts are deterministic.

| corpus, vocab | merges | relaxed rounds, no cap | closed rounds, no cap | mean batch | max | closed rounds, cap 256 | cap 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext.10M, 32k | 31,743 | 18,544 | 13,710 | 2.32 | 37 | 18,485 | 14,772 |
| tinystories.100M, 32k | 28,057 | 18,036 | 14,487 | 1.94 | 29 | 17,431 | 15,544 |
| tinystories.1G, 32k | 31,743 | 14,227 | 10,183 | 3.12 | 35 | 13,136 | 10,488 |
| wikitext.100M, 32k | 31,743 | 7,798 | 5,045 | 6.29 | 50 | 5,230 | 5,045 |
| **wikitext.full, 32k** | 31,743 | 3,409 | **2,561** | **12.39** | 59 | 2,561 | 2,561 |
| wikitext.full, 100k | 99,743 | 32,533 | 21,295 | 4.68 | 82 | 35,237 | 21,646 |

- The relaxed column reproduces the feasibility prototype's six figures exactly. The prototype's source was not kept.
- On `wikitext.full` v32k: strict needs 8,814 rounds, relaxed 3,409, closed form 2,561.
- **The cap matters where ties are large.** At the first planned default of 256, `wikitext.full` v100k takes 35,237
  rounds against 21,295 uncapped, which is worse than the relaxed rule without a cap. A cap of 1024 gives 21,646. The cap
  never affects exactness. B3 set the default to **4096**, which matches the uncapped rounds on both `wikitext.full`
  configurations (`tinystories.100M`: 14,561 against 14,487).
- **The tie scan stops early.** Growing T only lowers j, so k only shrinks, and never below max(g,1). Once k reaches that
  floor, the rest of T cannot change the answer. The batch is identical to a full scan.
- **Selection is the serial cost to watch.** Serial phase 3, oracle off, best of 3, on battery (indicative):
  `tinystories.1G` about 0.26 s against IncrSerial's 0.05 s; `wikitext.full` v100k about 1.5 s against 0.95 s;
  `wikitext.full` v32k about the same as IncrSerial (0.62–0.73 s against 0.74 s).

**Span and load, `wikitext.full`, v32k:**

| measure | one merge per step | batched rounds (closed) |
|---|---:|---:|
| share of tokens in steps ≥ 4096 tokens | 80.5% | 90.0% (relaxed: also 90.0%) |
| busiest owner / mean, T=5, contiguous | 2.30 | 2.08 |
| busiest owner / mean, T=5, hashed | 1.08 | 1.07 |
| busiest owner / mean, T=11, contiguous | 4.02 | 3.54 |
| busiest owner / mean, T=11, hashed | 1.18 | 1.16 |

Imbalance here is token work per owner on big steps, weighted by tokens (sum of maxima over sum of means). With 4T
hashed owners the per-owner figure is 1.24 at T=5, but a thread takes several owners under `schedule(dynamic,1)`, so that
is not a thread imbalance. The contiguous 2.30 and the 80.5% match the prototype. Two figures do not: hashed is 1.07–1.08
against the prototype's 1.02–1.06, and the batched share is 90.0% against its 91.2%. Neither difference is explained, and
the prototype's source is gone. B5 measured the imbalance in the real engine and got batchsim's numbers: 2.30 → 1.08 at
T=5 and 4.02 → 1.18 at T=11, one merge per step on `wikitext.full` v32k.

**A warning about what verification shows.** Two deliberately wrong rules (`all`, k = K; `one-based`, one member too many)
fail on `wikitext.10M` and `tinystories.100M`, but **both pass on `wikitext.full` v32k**, where big corpora have few
harmful ties. A clean `--verify-batches` on one large corpus therefore proves little by itself. The checks with teeth are
the constructed corpora, the small random alphabets and the smaller slices.

Earlier measurements, single runs on battery, indicative only: fork/join is about 50 µs per parallel step on macOS; merge
time tracks unique words, not corpus bytes (0.68 s at 540 MB, 0.73 s at 3.2 GB); at vocab 100k on `wikitext.full`,
IncrSerial phase 3 is 0.94 s and IncrPar 0.84 s, and phase 3 is about 37% of wall time. The estimate for phase 3 at 5
threads is about 1.8–2.2×, up from 1.14×. **That is an estimate, not a measurement.**

## Parallel rounds (B4)

Phase 3 on `wikitext.full`, median of 3, interleaved with IncrSerial, **on battery, with two stale ASan processes using
about 90% of a core in the background: indicative only, not the kill check.** `merges.txt` matched IncrSerial every time.

| threads, owners/thread, threshold | v32k | v100k |
|---|---:|---:|
| IncrSerial | 0.689 s | 1.019 s |
| IncrPar, 5 threads | 0.574 s (1.20×) | 0.896 s (1.14×) |
| incr-batch, 5, 1, 1024 | **0.383 s (1.80×)** | **0.722 s (1.41×)** |
| incr-batch, 5, 2, 1024 | 0.398 s (1.73×) | 0.792 s (1.29×) |
| incr-batch, 5, 4, 1024 | 0.425 s (1.62×) | 0.802 s (1.27×) |
| incr-batch, 11, 1, 1024 | 0.478 s (1.44×) | 0.897 s (1.14×) |

- Before the small-round skip, v100k at 5 threads was 0.79×: selection took 0.78 s and only 966 of 21,295 rounds went
  parallel.
- **Small-round skip (`batch_skip_small`, default on).** One merge is always exact, so a batch is only worth its selection
  cost if it can go parallel. A round is cut to one merge if its prefix can't reach `par_threshold_tokens`, or its best pair
  has under 1/8 of it. At 1 thread every round is one merge.
- **This separates two figures.** Skipping raises the reported rounds: on `wikitext.full` v32k, 19,888 at the default
  threshold against 2,561 with `--batch-full-rounds`. The span figures in this note are full-rounds figures. Don't quote a
  round count and a speedup from different modes. With full rounds and no skip, v32k measured 1.69× (5 threads, 1 owner per
  thread, threshold 4096).
- `owners_per_thread=1` beat 4 in every run here, including at 11 threads. Serial rounds pay one index lookup per owner per
  member. **The mains sweep confirmed it (B4, `results/b4_sweep.csv`): the default is now 1.**
- `tinystories.1G` v32k stays below break-even: 0.93× at 5 threads (0.059 s against 0.055 s). Ties keep batches small there,
  and phase 3 is about 3% of the run.
- The `last_touched` stamp is not needed for correctness in this engine: a rewritten word holds no member any more, so a
  second visit changes nothing. It only saves a rescan. The mutation that removes it survives for that reason.

## Design decisions

- **Keep the lazy max-heap**, not `std::set`: order updates run serially on the master, and that is the Amdahl term.
- **Selection** pops live entries, dropping duplicates, then builds the prefix, scans the tie group up to
  `batch_scan_cap`, and pushes back members k..K−1 and every scanned entry **before** applying.
- **`token_bytes` grows before any delta push**, because the heap comparator reads the new tokens' bytes (the S3 bug).
- **Hashed owners:** owner = mix64(w >> 4) % n, with n = `owners_per_thread`·T when T > 1 and 1 at T = 1, and
  `schedule(dynamic,1)`, so E-cores take smaller shares. Contiguous owners over the byte-sorted table cluster the words
  of a pair in one owner; that alone caps S6 near 2.2×.
- **IncrSerial stays the baseline.** It keeps its contiguous `begin/end` owner and its own code path.
- **Per-thread hot state** is `alignas(64)`; shards use the **low** hash bits (ankerl indexes by the high ones, a 51×
  slowdown in S4).
- `--verify-batches` is a shadow sequential engine on a copy of the table, sharing `merge_owner`.

## Risks

| risk | what happens then |
|---|---|
| Fork/join and serial order updates eat the gain | B4's kill check: < 1.3× at 5 threads at both vocabs stops the optimising; ship the theorem, the span reduction and the imbalance fix |
| Selection cost on large tie groups | `select_s` is logged per run; `batch_scan_cap` bounds it. It is 0.74–0.88 s at v100k, so parallel rounds must win that back; rounds too small to go parallel could skip the scan and take k=1, which is always exact |
| Mid-size words dominate rounds below the threshold | Those stay serial; `parallel_rounds` shows how many went parallel |
| Exactness bug that large corpora don't expose | Constructed corpora, small-alphabet random tests, mutation checks (see the warning above) |
| TSan was not checking anything | Its first CI run stopped at false positives inside libomp; fixed in B0 with `ignore_noninstrumented_modules=1` |
| TinyStories ties keep batches small | Reported as a limit: mean batch 1.9–3.1 there |
