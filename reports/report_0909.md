# CHIA atomic DRAM results — 9 September 2026

SimpleO3/DDR5 results for the final selected models from the completed
20-iteration Astra xhigh and Gemini 3.8 Flash campaigns. This is one campaign
per model with equal iteration counts, not an equal-compute or repeated-trial
study. ChampSim and gem5 generazibility results are still running.

## Architecture and settings

- **Atomic model:** C++20 initialization plus one immutable completion-cycle
  prediction per admitted request. Trusted code owns admission, mapping,
  capacities, and callbacks; no model tick, oracle state, or future-request
  access. Both campaigns start from the same fixed-delay seed.
- **Machine:** one four-wide SimpleO3 core, 128-entry instruction window;
  2 MB eight-way LLC, 64-byte lines, 47-cycle latency, 16 MSHRs.
  Frontend/memory clock ratios: 8/3. One 32-bit DDR5 channel, one rank,
  8 × 4 banks; `DDR5_16Gb_x8`, `DDR5_4800AN`.
- **Oracle:** GenericDDR with FRFCFS-RowHit, open rows, no refresh, 64-entry
  read/write buffers, and 0.5/0.8 write-drain watermarks. Candidate admission
  capacities match the oracle.
- **Runs:** eight training/eight disjoint test workloads; 20M instructions
  each, no separate warmup or trace wrap, full drain. Evaluation is closed-loop:
  predicted latency affects later arrivals. Builds use `-O3`; three workers
  per evaluation, at most twelve overall; 30-minute/4 GiB per-job guards.
- **Backends:** common CHIA sessions/tools, Ray jobs, and SQLite checkpoints;
  Codex `gpt-6-astra` at `xhigh`, Vertex `gemini-3.8-flash` at `high`.
  Campaign limit: 20 iterations, no harness output-token or dollar cap.
  Provider limits still apply.
- **Iteration:** continuous proposer session → edit/build/training diagnostics
  → one independent same-model/same-effort advisory review → revision → final
  evaluation → summary for the next iteration. Generic synthetic patterns,
  fixed-arrival open-loop replay, and trace inspection are available as
  diagnostics, not promotion datasets. Promotion requires mechanical validity,
  neither training objective worsening, and at least one improving, unrounded.
- **Separation and evidence:** agents see their model, API/task, own history,
  and training feedback—not other campaigns, the prior non-CHIA design, or test
  results. Native continuation, summaries, sources, and logs are retained with
  checksums and checked compression (gzip level 3). Testing makes no LLM calls.

Baselines are this repository's unchanged implementations: FixedLatency
(42-cycle base, bandwidth pipe), M/D/1 (10,000-cycle phase, 0.5 smoothing),
published channel WMG1 (10,000 ns window), and MeSS (fixed DDR5 curve,
1,000-access window, 0.05 convergence), not executions of the original projects.

## Metrics and aggregate comparison

Core error is `100 × mean_core(|C_model − C_oracle| / C_oracle)`, from final
per-core cycle statistics. Request latency is `depart − arrive` at the logical
LLC boundary in frontend cycles, including hits, merged misses, and miss owners.
Reads are paired by stable IDs; writes affect timing but are not scored.
With `Δ_i = latency_model,i − latency_oracle,i`, **L** is the full oracle mean
logical-read latency for that workload. MAE/L is `mean_i(|Δ_i|)/L`; absolute
drift/L is `|mean_i(Δ_i)|/L`. Headline values equally average eight workload
scores per split. Request ratios are dimensionless; lower is better.

Test request headlines temporarily waive the 10,000-owner-read screen.
All eight workloads retain equal weight and 100% bidirectional pairing;
complete-population checks remain enforced. Low-traffic cases are gcc
(5,177 owner reads), gromacs (8,283), and namd (3,563). Training/promotion
settings and original reports are unchanged. These are exploratory full-cohort
results, not memory-intensive-only scores.

| Model | Train core error (%) | Train MAE/L | Train abs. drift/L | Test core error (%) | Test MAE/L | Test abs. drift/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Astra xhigh — 20 iterations | 1.7876 | 0.1430 | 0.0264 | 1.9676 | 0.1576 | 0.0311 |
| Gemini 3.8 Flash high — 20 iterations | 2.1158 | 0.1336 | 0.0176 | 5.1583 | 0.1911 | 0.0897 |
| FixedLatency | 36.4036 | 0.3956 | 0.3568 | 30.7964 | 0.3930 | 0.3710 |
| M/D/1 | 40.2827 | 0.4339 | 0.4119 | 32.8718 | 0.4228 | 0.4140 |
| Channel WMG1 | 39.5204 | 0.4319 | 0.3978 | 32.7432 | 0.4232 | 0.4102 |
| MeSS | 28.0533 | 0.4916 | 0.3455 | 7.2916 | 0.3950 | 0.2333 |

## Per-workload breakdown

Each cell is **core error (%) / Request MAE/L**. Astra and Flash refer to the
same final 20-iteration selections as above. Both split memberships are
explicit in this table; no workload was selected or dropped after evaluation.

| Split | Workload | Astra xhigh | Gemini 3.8 Flash | FixedLatency | M/D/1 | Channel WMG1 | MeSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 401.bzip2 | 0.597 / 0.0404 | 0.036 / 0.0336 | 13.541 / 0.3031 | 14.499 / 0.3329 | 14.360 / 0.3278 | 10.194 / 0.2488 |
| Training | 429.mcf | 0.014 / 0.2853 | 1.363 / 0.2606 | 40.287 / 0.4023 | 44.817 / 0.4009 | 39.145 / 0.3946 | 13.273 / 0.3050 |
| Training | 434.zeusmp | 2.012 / 0.2267 | 0.010 / 0.2293 | 62.529 / 0.6068 | 62.645 / 0.6096 | 62.630 / 0.6095 | 13.245 / 0.2747 |
| Training | 436.cactusADM | 7.541 / 0.0580 | 0.019 / 0.0275 | 58.204 / 0.7876 | 62.918 / 0.8490 | 62.894 / 0.8489 | 38.398 / 0.7574 |
| Training | 462.libquantum | 0.087 / 0.0012 | 0.084 / 0.0012 | 6.235 / 0.0657 | 1.782 / 0.1067 | 1.779 / 0.1067 | 97.649 / 1.0281 |
| Training | 482.sphinx3 | 1.802 / 0.1515 | 7.796 / 0.1253 | 25.215 / 0.2305 | 33.246 / 0.2814 | 33.221 / 0.2801 | 32.484 / 0.6633 |
| Training | 483.xalancbmk | 0.194 / 0.0976 | 0.423 / 0.0743 | 33.590 / 0.3360 | 43.725 / 0.3983 | 43.671 / 0.3979 | 0.054 / 0.3355 |
| Training | 519.lbm | 2.053 / 0.2837 | 7.195 / 0.3171 | 51.629 / 0.4329 | 58.630 / 0.4928 | 58.464 / 0.4893 | 19.130 / 0.3202 |
| Test | 403.gcc | 0.010 / 0.0261 | 0.032 / 0.0131 | 2.460 / 0.2554 | 3.014 / 0.3165 | 3.014 / 0.3165 | 3.256 / 0.5461 |
| Test | 433.milc | 7.598 / 0.3862 | 9.229 / 0.3759 | 58.972 / 0.5856 | 61.606 / 0.6022 | 61.601 / 0.6019 | 18.732 / 0.4135 |
| Test | 435.gromacs | 0.098 / 0.0063 | 0.098 / 0.0058 | 6.851 / 0.2873 | 7.142 / 0.2982 | 7.142 / 0.2982 | 12.604 / 0.5061 |
| Test | 444.namd | 0.002 / 0.0085 | 0.007 / 0.0085 | 1.207 / 0.2417 | 1.466 / 0.2973 | 1.465 / 0.2974 | 4.023 / 0.5237 |
| Test | 450.soplex | 1.719 / 0.4237 | 13.690 / 0.7984 | 44.124 / 0.4547 | 48.263 / 0.4830 | 47.506 / 0.4899 | 1.813 / 0.4796 |
| Test | 459.GemsFDTD | 0.909 / 0.0236 | 0.669 / 0.0152 | 44.136 / 0.4703 | 47.956 / 0.5151 | 47.906 / 0.5145 | 5.036 / 0.1324 |
| Test | 471.omnetpp | 1.470 / 0.0582 | 0.100 / 0.0440 | 37.691 / 0.4382 | 36.554 / 0.4246 | 36.335 / 0.4218 | 1.922 / 0.1040 |
| Test | 549.fotonik3d | 3.934 / 0.3285 | 17.440 / 0.2674 | 50.931 / 0.4111 | 56.973 / 0.4456 | 56.976 / 0.4456 | 10.948 / 0.4544 |

On the full cohorts, both learned models improve both aggregate objectives over every baseline.
Flash has lower training request error; Astra has lower test errors. Flash's
largest test core errors are fotonik3d and soplex. Tails remain substantial:
soplex's `P99(|Δ|)/L` is 4.947 for Astra versus 12.734 for Flash, with signed
extremes of −9.992/+3.730 L and −10.607/+17.120 L, respectively.

This remains a small pilot split: the three sub-10,000-read cases are all in
test, with none in training. Aggregate rankings are sensitive to the workload
mix; these results do not establish performance across all memory behaviors.

## Evidence

Flash's final-model evaluation completed all 48 native jobs: eight test
workloads × oracle, candidate, and four baselines. All four baseline reports
match Astra's reports exactly on both training and test. Candidate source IDs
are `18d1dc597295…` (Astra) and `6d1190d76d4a…` (Flash), both after iteration 20.

- [Astra measurements and training results](/home/dev/chia-simpleo3-tests-20260909/astra_xhigh/report.json);
  [exploratory test aggregates](/home/dev/chia-simpleo3-tests-20260909/astra_xhigh/exploratory_ungated/report.json).
- [Flash final measurements and training results](/home/dev/chia-simpleo3-tests-20260909/gemini38flash/report.json);
  [exploratory test aggregates](/home/dev/chia-simpleo3-tests-20260909/gemini38flash/exploratory_ungated/report.json).
- [Evaluation configuration](../tools/chia_loop/configs/ddr5_public_transfer_v2.json),
  [shared SimpleO3 settings](../tools/eval/simpleo3.py), and
  [CHIA workflow](../tools/chia_loop/framework/campaign.py).
