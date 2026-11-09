# Third-party notices

The original code in this repository is licensed under Apache-2.0 (see [LICENSE](LICENSE)). The components below keep their own licenses, and individual source files keep any license headers they carry.

| Component | Location | License |
|---|---|---|
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and its bundled projects | `training/nemo_rl/` | See `training/nemo_rl/LICENSE` and the license files in its subfolders |
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | `evaluation_qwen35/flashrag/`, `evaluation_search_r1/flashrag/`, `local_retriever/flashrag/` | MIT, reproduced below |
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Adapted in the search and reward logic | Apache-2.0 |

Benchmark datasets in `data/` and the base model weights remain under their original terms.

## FlashRAG license

```text
MIT License

Copyright (c) 2024 Jiajie Jin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
