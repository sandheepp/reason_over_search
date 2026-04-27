# Reason Over Search

## Papers

1. [ReSearch](https://www.alphaxiv.org/abs/2503.19470)
2. [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

## Milestone 1: Baseline [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

### Phase 1 Context

- Retriever code is working and has already been tested.
- Evaluation code is a hybrid:
  - base evaluation flow adapted from ReSearch
  - Search-R1 prompts and evaluation logic added on top
- Search-R1 does not provide an official evaluation pipeline, so this adapted pipeline must be validated carefully.

### Goal

Reproduce the Search-R1 3B baseline for both model variants:
- base
- instruct

Target benchmarks:
- Bamboogle (first dataset for fast iteration)
- 2WikiMultiHopQA
- TriviaQA
- PopQA
- MuSiQue
- NQ (in-distribution)
- HotpotQA (in-distribution)

### Dataset Requirements

- Download missing datasets from Hugging Face.
- Use the Search-R1 paper to confirm official dataset sources.
- Verify existing local datasets against online sources.
- Confirm correct split per benchmark (test/dev/train), including whether `dev` is treated as evaluation where no test split is available.
- Use AlphaXiv AI paper assistant when split definitions are ambiguous.

### Reproducibility and Cost Constraints

- Primary objective: reproducible setup on Vast.ai and in-house GPUs.
- Secondary objective: minimize cost and wall-clock runtime.
- Track both reproducibility and cost clearly; these are important for publication quality.

### Step-by-step

1. Prepare runtime on Vast.ai.
   - Use this Docker image: [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1)
   - Or build and push your own image using `docker/reason-over-search-v1/README.md`.
   - Create a Vast.ai custom template from that image and start an instance.
   - Storage guidance: allocate at least `150 GB` for indexes and model files.
   - GPU guidance: start testing with lower-cost options first (for example, 40 GB VRAM class) before moving to larger GPUs (for example, A100 80 GB), depending on throughput and stability.

2. Set up local retriever on the Vast instance.
   - Follow `local_retriever/README.md`.
   - If you are using the Docker image above, required environments are already installed, so you can skip environment creation.
   - Download:
     - corpus
     - indexes
     - embedding model

3. Run and validate the retriever service.
   - Start the retriever using `local_retriever/README.md`.
   - Run the health/search test calls to confirm it is serving correctly.
   - Resource note: the retriever runs on CPU and needs about `~80 GB` CPU RAM.

4. Set up evaluation models and SGLang on the Vast instance.
   - Go to `evaluation_search_r1/`.
   - Activate the evaluation environment.
   - Download both Search-R1 model variants (base and instruct).
   - Start the SGLang server for the model being evaluated.

5. Run baseline evaluations.
   - Run all target benchmarks listed in this milestone.
   - Run each benchmark `5` times per model variant.
   - Report the average score across the 5 runs for each benchmark.

### If Results Diverge from Paper

- Inspect prompt templates and answer verification logic in `evaluation_search_r1/flashrag/search_r1/`.
- Verify behavior separately for base and instruct variants.
- Cross-check implementation details against the official repository:
  - [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1)

### Deliverables

1. All evaluation datasets downloaded, verified, and tracked appropriately.
2. Evaluation pipeline validated and run on all listed benchmarks.
3. Aggregated results from repeated runs (5 runs per benchmark/model variant).
4. Code reviewed and cleaned up for clarity (remove unnecessary complexity/fluff).
5. Docker setup verified; documentation updated for clear reproduction.
6. Repository state suitable for easy reproduction and publication submission.
