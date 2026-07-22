# Reference Repositories

## Natural Language Autoencoders

- Upstream: `https://github.com/kitft/natural_language_autoencoders.git`
- Vendored path: `external/natural_language_autoencoders`
- Vendored commit: `047eb8e40452982d38f83721f9fb2c77baf6b0cf`
- Vendored on: `2026-05-13`

Use this repository as a reference for:

- input-embedding injection,
- sidecar metadata conventions,
- normalized MSE / cosine reconstruction scoring,
- AR critic/value-head construction,
- Qwen/Gemma released-checkpoint QC.

Do not assume its model-wrapper assumptions apply directly to Nano. Nano requires adapters for `.backbone`, `.backbone.layers`, `.backbone.norm_f`, `.backbone.embeddings`, chat-template control, and hybrid Mamba/attention cache behavior.
