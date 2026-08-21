# Patches

The checked-out `source/` tree is the deployment source. Keep only patches
that are still needed to reproduce a build or port current work.

## Prime-RL

- `prime-rl/runtime-deepep-score.patch` is applied by the current Dockerfile.
- `prime-rl/v0.8.0-glm52/` is the ordered GLM-5.2 port. Apply its numbered
  files with `git am`. Its `SHA256SUMS` file identifies each file.

## Dependencies

- `renderers/v0.8.0-disconnected-and-tokenizer.patch`
- `verifiers/v0.8.0-disconnected-provider-retry.patch`

These dependency patches apply only to the revisions named in
`prime-rl/v0.8.0-glm52/README.md`.

Before a patch update:

```bash
cd patches/prime-rl/v0.8.0-glm52
sha256sum -c SHA256SUMS
```

Update the changed patch and `SHA256SUMS` in the same change.
