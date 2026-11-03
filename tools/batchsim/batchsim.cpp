// batchsim: standalone cross-check of the batched-merging rule (docs/notes/batched_merging.md).
// Replays exact BPE from a words.tsv with a std::set order (no project code on purpose, and not in
// the CMake build). Each round computes a batch from the live order, then applies its members one
// merge at a time and asserts that each one is the sequential argmax at that moment.
//
//   c++ -std=c++20 -O2 tools/batchsim/batchsim.cpp -o build/batchsim
//   build/batchsim --words runs/x/words.tsv --vocab-size 32000 [--rule closed] [--cap 256]
//                  [--specials 1] [--threads 5] [--big 4096] [--dump-merges FILE]
//
// Rules, with 0-based j = the first member whose tokens appear in T (j = K if none):
//   strict     k = K if T is empty, else max(g, 1)
//   relaxed    k = max(min(K, j), g, 1)
//   closed     k = max(min(K, j + 1), g, 1)          the rule the engine ships
//   one-based  k = max(min(K, j + 2), g, 1)          WRONG on purpose: the assertion must fire
//   all        k = K                                 WRONG on purpose: the assertion must fire
// An incomplete tie scan (cap reached) gives k = max(g, 1) for every rule but `all`.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Pair = uint64_t;
Pair make_pair_key(uint32_t a, uint32_t b) {
  return (Pair(a) << 32) | b;
}
uint32_t first(Pair p) {
  return uint32_t(p >> 32);
}
uint32_t second(Pair p) {
  return uint32_t(p);
}

std::vector<std::string> g_bytes;  // token id -> bytes

struct Entry {
  int64_t count;
  Pair pair;
};

// Best first: count, then the (bytes(a), bytes(b)) tuple (SPEC.md §4). The pair id is the last key
// so the set never drops a distinct pair whose tokens happen to have the same bytes.
struct BestFirst {
  bool operator()(const Entry& x, const Entry& y) const {
    if (x.count != y.count) return x.count > y.count;
    const std::string_view xa = g_bytes[first(x.pair)], ya = g_bytes[first(y.pair)];
    if (const int c = xa.compare(ya); c != 0) return c > 0;
    const std::string_view xb = g_bytes[second(x.pair)], yb = g_bytes[second(y.pair)];
    if (const int c = xb.compare(yb); c != 0) return c > 0;
    return x.pair > y.pair;
  }
};

struct Word {
  std::vector<uint32_t> tok;
  int64_t freq;
};

uint64_t mix64(uint64_t x) {  // splitmix64 finalizer
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Contiguous owners balanced by token count, the same rule as partition_words in the engine.
std::vector<uint8_t> contiguous_owners(const std::vector<Word>& words, size_t n) {
  std::vector<uint8_t> owner(words.size());
  size_t total = 0, seen = 0, w = 0;
  for (const Word& word : words) total += word.tok.size();
  for (size_t o = 0; o < n; ++o) {
    const size_t target = total * (o + 1) / n;
    while (w < words.size() && (seen < target || o + 1 == n)) {
      seen += words[w].tok.size();
      owner[w++] = uint8_t(o);
    }
  }
  return owner;
}

// Work of big steps: busiest owner / mean owner, weighted by tokens (sum of maxima / sum of means).
struct Imbalance {
  double busiest = 0, mean = 0;
  void add(const std::vector<uint64_t>& load) {
    uint64_t sum = 0, top = 0;
    for (const uint64_t l : load) {
      sum += l;
      top = std::max(top, l);
    }
    busiest += double(top);
    mean += double(sum) / double(load.size());
  }
  double ratio() const { return mean > 0 ? busiest / mean : 0; }
};

std::string gpt2_unicode(unsigned char b) {
  static const std::array<int, 256> table = [] {
    std::array<int, 256> t{};
    int n = 0;
    for (int c = 0; c < 256; ++c) {
      const bool printable = (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || c >= 0xAE;
      t[c] = printable ? c : 256 + n++;
    }
    return t;
  }();
  const int cp = table[b];
  std::string s;
  if (cp < 0x80) {
    s += char(cp);
  } else {
    s += char(0xC0 | (cp >> 6));
    s += char(0x80 | (cp & 0x3F));
  }
  return s;
}

std::string encoded(const std::string& bytes) {
  std::string s;
  for (const unsigned char c : bytes) s += gpt2_unicode(c);
  return s;
}

std::vector<Word> read_words(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "cannot read " << path << "\n";
    std::exit(2);
  }
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  std::vector<Word> words;
  std::string line;
  while (std::getline(in, line)) {
    const size_t tab = line.find('\t');
    Word w;
    for (size_t i = 0; i + 1 < tab; i += 2) {
      w.tok.push_back(uint32_t(nibble(line[i]) * 16 + nibble(line[i + 1])));
    }
    w.freq = std::stoll(line.substr(tab + 1));
    words.push_back(std::move(w));
  }
  return words;
}

struct Options {
  std::string words_path, rule = "closed", dump_path;
  size_t vocab_size = 32000, specials = 1, cap = SIZE_MAX, threads = 5, big = 4096;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--words")
      o.words_path = v;
    else if (k == "--rule")
      o.rule = v;
    else if (k == "--dump-merges")
      o.dump_path = v;
    else if (k == "--vocab-size")
      o.vocab_size = std::stoull(v);
    else if (k == "--specials")
      o.specials = std::stoull(v);
    else if (k == "--cap")
      o.cap = std::stoull(v);
    else if (k == "--threads")
      o.threads = std::stoull(v);
    else if (k == "--big")
      o.big = std::stoull(v);
    else {
      std::cerr << "unknown option " << k << "\n";
      std::exit(2);
    }
  }
  if (o.words_path.empty()) {
    std::cerr << "usage: batchsim --words words.tsv --vocab-size N [--rule R] [--cap N]\n";
    std::exit(2);
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  std::vector<Word> words = read_words(opt.words_path);

  for (int i = 0; i < 256; ++i) g_bytes.emplace_back(1, char(i));
  for (size_t i = 0; i < opt.specials; ++i) g_bytes.emplace_back("<special>");
  const size_t budget = opt.vocab_size - g_bytes.size();

  std::unordered_map<Pair, int64_t> counts;
  std::unordered_map<Pair, std::vector<uint32_t>> index;
  for (uint32_t w = 0; w < words.size(); ++w) {
    const auto& t = words[w].tok;
    for (size_t i = 0; i + 1 < t.size(); ++i) {
      const Pair p = make_pair_key(t[i], t[i + 1]);
      counts[p] += words[w].freq;
      auto& list = index[p];
      if (list.empty() || list.back() != w) list.push_back(w);
    }
  }
  std::set<Entry, BestFirst> order;
  for (const auto& [p, c] : counts) order.insert({c, p});

  // Owner maps for the imbalance report: T contiguous, T hashed, 4T hashed.
  const size_t T = opt.threads;
  const std::vector<uint8_t> own_contig = contiguous_owners(words, T);
  const auto hashed = [](uint32_t w, size_t n) { return size_t(mix64(w >> 4) % n); };
  Imbalance imb_contig, imb_hash_t, imb_hash_4t;
  Imbalance imb_contig_step, imb_hash_step;
  uint64_t tokens_all = 0, tokens_big_steps = 0, tokens_big_rounds = 0;
  size_t big_rounds = 0, big_steps = 0;

  std::vector<uint32_t> touched(words.size(), UINT32_MAX);
  std::vector<uint32_t> touched_round(words.size(), UINT32_MAX);
  std::vector<uint32_t> tok_mark;
  std::vector<std::pair<uint32_t, uint32_t>> merges;
  std::vector<size_t> hist(9, 0);  // 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129+
  std::array<size_t, 5> stops{};   // disjoint, tie, self, budget, capped
  size_t rounds = 0, max_batch = 0, max_prefix = 0, incomplete = 0, tail_max = 0, tail_sum = 0;

  while (merges.size() < budget && !order.empty()) {
    // Prefix p[0..K): best-first, pairwise token-disjoint; a self-pair may only be last.
    tok_mark.assign(g_bytes.size(), 0);
    std::vector<Entry> prefix;
    const size_t left = budget - merges.size();
    auto it = order.begin();
    int stop = 0;
    while (it != order.end()) {
      if (prefix.size() == left) {
        stop = 3;
        break;
      }
      const uint32_t a = first(it->pair), b = second(it->pair);
      if (tok_mark[a] || tok_mark[b]) {
        stop = 0;
        break;
      }
      prefix.push_back(*it);
      tok_mark[a] = tok_mark[b] = 1;
      ++it;
      if (a == b) {
        stop = 2;
        break;
      }
    }
    const size_t K = prefix.size();
    const int64_t c_star = prefix.back().count;

    // T: every later live pair with count c*, starting at the entry that stopped the prefix.
    std::vector<uint32_t> in_t(g_bytes.size(), 0);
    bool complete = true;
    size_t scanned = 0;
    for (; it != order.end() && it->count == c_star; ++it) {
      if (scanned == opt.cap) {
        complete = false;
        break;
      }
      in_t[first(it->pair)] = in_t[second(it->pair)] = 1;
      ++scanned;
    }
    size_t g = 0, j = K;
    for (size_t i = 0; i < K; ++i) {
      if (prefix[i].count > c_star) ++g;
      if (j == K && (in_t[first(prefix[i].pair)] || in_t[second(prefix[i].pair)])) j = i;
    }

    size_t k;
    if (opt.rule == "all") {
      k = K;
    } else if (!complete) {
      k = std::max<size_t>(g, 1);
      ++incomplete;
    } else if (opt.rule == "strict") {
      k = scanned == 0 ? K : std::max<size_t>(g, 1);
    } else if (opt.rule == "relaxed") {
      k = std::max({std::min(K, j), g, size_t(1)});
    } else if (opt.rule == "closed") {
      k = std::max({std::min(K, j + 1), g, size_t(1)});
    } else if (opt.rule == "one-based") {
      k = std::max({std::min(K, j + 2), g, size_t(1)});
    } else {
      std::cerr << "unknown rule " << opt.rule << "\n";
      return 2;
    }

    if (k < K) stop = complete ? 1 : 4;
    ++stops[stop];
    tail_max = std::max(tail_max, scanned);
    tail_sum += scanned;

    // Apply the members one merge at a time; each must be the sequential argmax right now.
    std::vector<uint64_t> load_contig(T, 0), load_hash_t(T, 0), load_hash_4t(4 * T, 0);
    uint64_t round_tokens = 0;
    for (size_t i = 0; i < k; ++i) {
      const Entry& member = prefix[i];
      const Entry top = *order.begin();
      if (top.pair != member.pair) {
        std::cerr << "VIOLATION round " << rounds << " slot " << i << " of " << k << " (K=" << K
                  << " g=" << g << " j=" << j << "): batch has (" << first(member.pair) << ","
                  << second(member.pair) << ") count " << member.count << ", sequential argmax ("
                  << first(top.pair) << "," << second(top.pair) << ") count " << top.count << "\n";
        return 1;
      }
      const uint32_t a = first(member.pair), b = second(member.pair);
      const uint32_t z = uint32_t(g_bytes.size());
      g_bytes.push_back(g_bytes[a] + g_bytes[b]);
      merges.emplace_back(a, b);

      const uint32_t m = uint32_t(merges.size());
      const std::vector<uint32_t> list = std::move(index[member.pair]);
      index.erase(member.pair);
      std::unordered_map<Pair, int64_t> delta;
      std::vector<uint64_t> step_contig(T, 0), step_hash(T, 0);
      uint64_t step_tokens = 0;
      for (const uint32_t w : list) {
        if (touched[w] == m) continue;
        touched[w] = m;
        auto& t = words[w].tok;
        bool has = false;
        for (size_t x = 0; x + 1 < t.size(); ++x) has |= t[x] == a && t[x + 1] == b;
        if (!has) continue;

        const uint64_t len = t.size();
        step_tokens += len;
        step_contig[own_contig[w]] += len;
        step_hash[hashed(w, T)] += len;
        if (touched_round[w] != rounds) {  // one whole-word diff per word per round in the engine
          touched_round[w] = uint32_t(rounds);
          round_tokens += len;
          load_contig[own_contig[w]] += len;
          load_hash_t[hashed(w, T)] += len;
          load_hash_4t[hashed(w, 4 * T)] += len;
        }

        const int64_t f = words[w].freq;
        for (size_t x = 0; x + 1 < t.size(); ++x) delta[make_pair_key(t[x], t[x + 1])] -= f;
        size_t out = 0;
        for (size_t x = 0; x < t.size();) {
          if (x + 1 < t.size() && t[x] == a && t[x + 1] == b) {
            t[out++] = z;
            x += 2;
          } else {
            t[out++] = t[x++];
          }
        }
        t.resize(out);
        for (size_t x = 0; x + 1 < t.size(); ++x) {
          const Pair p = make_pair_key(t[x], t[x + 1]);
          delta[p] += f;
          if (t[x] == z || t[x + 1] == z) {
            auto& l = index[p];
            if (l.empty() || l.back() != w) l.push_back(w);
          }
        }
      }
      for (const auto& [p, d] : delta) {
        if (d == 0) continue;
        auto cit = counts.find(p);
        if (cit != counts.end()) order.erase({cit->second, p});
        const int64_t c = (cit == counts.end() ? 0 : cit->second) + d;
        if (c < 0) {
          std::cerr << "negative count\n";
          return 1;
        }
        if (c == 0) {
          counts.erase(p);
        } else {
          counts[p] = c;
          order.insert({c, p});
        }
      }
      tokens_all += step_tokens;
      if (step_tokens >= opt.big) {
        ++big_steps;
        tokens_big_steps += step_tokens;
        imb_contig_step.add(step_contig);
        imb_hash_step.add(step_hash);
      }
    }
    if (round_tokens >= opt.big) {
      ++big_rounds;
      tokens_big_rounds += round_tokens;
      imb_contig.add(load_contig);
      imb_hash_t.add(load_hash_t);
      imb_hash_4t.add(load_hash_4t);
    }

    ++rounds;
    max_batch = std::max(max_batch, k);
    max_prefix = std::max(max_prefix, K);
    size_t bucket = 0;
    while (bucket < 8 && k > (size_t(1) << bucket)) ++bucket;
    ++hist[bucket];
  }

  if (!opt.dump_path.empty()) {
    std::ofstream out(opt.dump_path, std::ios::binary);
    out << "#version: 0.2\n";
    for (const auto& [a, b] : merges)
      out << encoded(g_bytes[a]) << ' ' << encoded(g_bytes[b]) << '\n';
  }

  const size_t n = merges.size();
  std::printf("rule=%s cap=%s merges=%zu rounds=%zu mean_batch=%.2f max_batch=%zu max_prefix=%zu\n",
              opt.rule.c_str(), opt.cap == SIZE_MAX ? "none" : std::to_string(opt.cap).c_str(), n,
              rounds, rounds ? double(n) / double(rounds) : 0.0, max_batch, max_prefix);
  std::printf("every member verified as the sequential argmax: %zu merges, 0 violations\n", n);
  static const char* labels[9] = {"1",     "2",     "3-4",    "5-8", "9-16",
                                  "17-32", "33-64", "65-128", "129+"};
  std::printf("batch sizes:");
  for (size_t i = 0; i < 9; ++i) std::printf(" %s:%zu", labels[i], hist[i]);
  std::printf("\nstops: disjoint:%zu tie:%zu self:%zu budget:%zu capped:%zu\n", stops[0], stops[1],
              stops[2], stops[3], stops[4]);
  std::printf("tie scan: max %zu, mean %.1f, incomplete rounds %zu\n", tail_max,
              rounds ? double(tail_sum) / double(rounds) : 0.0, incomplete);
  const double all = tokens_all ? double(tokens_all) : 1.0;
  std::printf(
      "tokens in steps >= %zu: one merge per step %.1f%% (%zu steps), batched rounds %.1f%% "
      "(%zu rounds)\n",
      opt.big, 100.0 * double(tokens_big_steps) / all, big_steps,
      100.0 * double(tokens_big_rounds) / all, big_rounds);
  std::printf("busiest/mean at T=%zu, one merge per step: contiguous %.2f, hashed %.2f\n", T,
              imb_contig_step.ratio(), imb_hash_step.ratio());
  std::printf(
      "busiest/mean at T=%zu, batched rounds: contiguous %.2f, hashed(T) %.2f, "
      "hashed(4T) %.2f\n",
      T, imb_contig.ratio(), imb_hash_t.ratio(), imb_hash_4t.ratio());
  return 0;
}
