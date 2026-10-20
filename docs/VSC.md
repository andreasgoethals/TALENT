# TALENT on VSC

Shared infrastructure reference: [VSC Library](../vsc-library/README.md).
Initialize the pinned checkout with `git submodule update --init -- vsc-library`.
Use `git submodule status -- vsc-library` to inspect the current committed reference.
Its source snapshot, guides and Slurm templates are read-only. Keep this fork's
environment, account, resource choices and operating commands in this document.

The library registration does not establish a tested VSC deployment. Before
running experiments, review the library's upstream-change flags and verify the
site-specific environment and bounded pilot. No job was submitted in this migration.

Canonical maintenance and published-pin propagation run from the separate
`0. Literature` workspace. This fork consumes VSC only; TFM and DM are optional.
