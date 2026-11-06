#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <map>

#include "parabpe/corpus.hpp"
#include "parabpe/io.hpp"
#include "parabpe/trainer.hpp"

using namespace parabpe;

namespace {

int run_train(const std::string& input, const std::string& out_dir, const TrainConfig& cfg,
              bool dump_words) {
  std::filesystem::create_directories(out_dir);
  if (dump_words) {  // phase 1 only: lets S1/S4 be checked without the merge engines
    const Corpus corpus = Corpus::open(input);
    PhaseTimings t;
    const WordCounts counts = count_words(corpus.text(), cfg, &t);
    write_file(out_dir + "/words.tsv", format_words(counts));
    std::cout << "words=" << counts.words.size() << " pretok=" << t.pretok << "s\n";
    return 0;
  }

  const TrainResult r = train(input, cfg);
  write_file(out_dir + "/merges.txt", format_merges(r.merges, r.vocab));
  write_file(out_dir + "/vocab.json", format_vocab(r.vocab, cfg.special_tokens.size()));
  write_file(out_dir + "/timings.json", format_timings(r, input, cfg));
  std::cout << "merges=" << r.merges.size() << " total=" << r.timings.total << "s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"ParaBPE: parallel byte-level BPE tokenizer training"};
  app.require_subcommand(1);

  TrainConfig cfg;
  std::string input, out_dir = "runs/latest";
  std::vector<std::string> specials;
  bool no_special = false, dump_words = false;

  const std::map<std::string, PretokStrategy> pretok_map{
      {"serial", PretokStrategy::Serial}, {"omp", PretokStrategy::OMP}};
  const std::map<std::string, PretokReduce> reduce_map{
      {"bucketed", PretokReduce::Bucketed}, {"critical", PretokReduce::Critical}};
  const std::map<std::string, MergeStrategy> merge_map{
      {"naive-serial", MergeStrategy::NaiveSerial},
      {"incr-serial", MergeStrategy::IncrSerial},
      {"naive-par", MergeStrategy::NaivePar},
      {"incr-par", MergeStrategy::IncrPar},
      {"incr-batch", MergeStrategy::IncrBatch}};

  const std::map<std::string, OwnerMap> owner_map{{"contiguous", OwnerMap::Contiguous},
                                                  {"hashed", OwnerMap::Hashed}};

  auto* tr = app.add_subcommand("train", "Train merges on a prepared corpus");
  tr->add_option("--input", input, "Prepared corpus (UTF-8, <|endoftext|>-separated)")
      ->required()
      ->check(CLI::ExistingFile);
  tr->add_option("--out", out_dir, "Output directory");
  tr->add_option("--vocab-size", cfg.vocab_size)->check(CLI::Range(257, 1 << 20));
  tr->add_option("--threads", cfg.threads)->check(CLI::PositiveNumber);
  tr->add_option("--pretok", cfg.pretok)->transform(CLI::CheckedTransformer(pretok_map));
  tr->add_option("--pretok-reduce", cfg.pretok_reduce,
                 "OMP pre-tokenizer merge: bucketed (default) or critical")
      ->transform(CLI::CheckedTransformer(reduce_map));
  tr->add_option("--merge", cfg.merge)->transform(CLI::CheckedTransformer(merge_map));
  tr->add_option("--par-threshold-tokens", cfg.par_threshold_tokens);
  tr->add_option("--owner-map", cfg.owner_map, "incr-par: how words are assigned to owners")
      ->transform(CLI::CheckedTransformer(owner_map));
  tr->add_flag("--owner-stats", cfg.owner_stats,
               "incr-par: report busiest/mean owner load (slows the run; not for timings)");
  tr->add_option("--work-unit-bytes", cfg.work_unit_bytes);
  tr->add_flag("--verify-batches", cfg.verify_batches,
               "incr-batch: check every member against a sequential oracle");
  tr->add_option("--owners-per-thread", cfg.owners_per_thread,
                 "incr-batch: hashed owners per thread")
      ->check(CLI::PositiveNumber);
  tr->add_option("--batch-scan-cap", cfg.batch_scan_cap,
                 "incr-batch: tie-group entries scanned per round");
  tr->add_flag("!--batch-full-rounds", cfg.batch_skip_small,
               "incr-batch: batch every round, even ones too small to go parallel (span figures)");
  tr->add_option("--trace", cfg.trace_path, "Write a Chrome Trace JSON file");
  auto* special_opt = tr->add_option("--special", specials,
                                     "Special token, repeatable (default: <|endoftext|>)")
                          ->allow_extra_args(false);
  tr->add_flag("--no-special", no_special, "Train with no special tokens")->excludes(special_opt);
  tr->add_flag("--dump-words", dump_words, "Write <out>/words.tsv after phase 1 and stop");

  auto* enc = app.add_subcommand("encode", "Encode text with trained merges (S2+)");
  std::string model_dir, enc_input, enc_out;
  enc->add_option("--model", model_dir)->required();
  enc->add_option("--input", enc_input)->required();
  enc->add_option("--out", enc_out)->required();

  CLI11_PARSE(app, argc, argv);

  try {
    if (*tr) {
      // Same rule as bpe_ref.py: the default special applies only when none is given.
      if (no_special) cfg.special_tokens.clear();
      if (!specials.empty()) cfg.special_tokens = specials;
      return run_train(input, out_dir, cfg, dump_words);
    }
    if (*enc) {
      std::cerr << "encode: not implemented yet\n";
      return 2;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
