# Shared datasets

This project publishes frozen data used by benchmarking and remote fine-tuning.

- Use Python 3.12. The core release format and publisher have no inference dependencies.
- Put reusable code in `src/thesis_datasets/`; retain existing translation pipelines.
- Preserve original IDs, text, targets, order, and split membership when publishing.
- Source revision and prepared version are separate. Never edit an existing release.
- Downloaded sources, translations, releases, and runtime state remain ignored.
- Do not run translation requests or regenerate existing samples unless requested.
- Test checksums, target mappings, incomplete exports, and reproducible publication.
- Document setup and commands in the main README. Do not edit the vault without authorization.
