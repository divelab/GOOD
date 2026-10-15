# Hugging Face dataset download migration

Implementation checkpoint: 2026-10-14 20:41 PDT (2026-10-15 03:41 UTC).
Worktree: `GOOD-hf-downloads`; branch: `codex/huggingface-downloads`;
base: `b53566c9297bc65b90a7f2213fb9ffa930f5b6e5`.

## Scope and interface

All 13 existing archive-backed loaders call
`download_dataset(name, root, required_files)`. Their constructors, processed
filenames, split selection, tensor deserialization, and generation code retain
existing behavior. The original ZINC CSV helper remains part of generation and
continues to use its upstream source.

The shared module owns Hugging Face transport, archive validation, file locking,
staging, and installation. It uses `citrinegui/GOOD` dataset archives named after
the existing classes. Training callers supply the processed files required by
PyG. A complete local dataset returns before contacting Hugging Face; an
incomplete directory is repaired by installing only missing files. Existing
files, including locally generated files and unrelated domains, are preserved.

Hugging Face owns the resumable archive cache. One file lock per dataset root and
class serializes concurrent downloads/extractions. Archives are extracted in a
same-filesystem temporary directory, checked for expected processed files, then
missing files are installed by atomic rename. Unsafe archive paths and symlinks
are rejected. Failed extraction cannot publish partial files; interrupted
installation can leave complete individual files, which the next call reuses.
Exceptions propagate to training without falling back to another source or
triggering regeneration.

Peak disk use includes the Hub's compressed archive and an extracted staging
copy. Installation moves staged files rather than duplicating their bytes.
Existing partial files are recognized by existence, as in the previous loader;
this change does not detect previously corrupted tensor files. No throughput or
whole-training performance claim is made.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test/data/test_dataset_download.py`:
  **42 passed** in 0.22 seconds on Python 3.11.5 / pytest 7.4.0.
- All changed download modules, the 13 callers, original ZINC helper, package
  initializer and new test file parse with Python 3.8 grammar: **17 files**.
- `git diff --check`: passed.
- The tests use fixture ZIPs and a stub Hub transport. They execute the actual
  download methods from every loader while omitting heavyweight imports. They
  cover byte/layout preservation, repeated offline cache use, incomplete cache
  repair, generation mode, errors, unsafe archives, symlink redirects, interrupted
  installation, and concurrent callers.
- This checkpoint does not claim real PyTorch/PyG training, live Hub delivery, or
  original archive parity. Those checks belong to the separate mirror and runtime
  validation work and must be recorded before final delivery.

## Skills and clean-code review

Task/stack discovery used the existing local `find-skills` method for Python
package integration and filesystem/download testing. Local methods were
sufficient; no worker-installed skill was needed. The coordinator's separate
Hugging Face discovery found the Skills CLI unavailable due to a missing `yaml`
module, and used official Hugging Face documentation instead.

Read local skill files from the installed skills directory; content versions
are SHA-256 identifiers, not assumed upstream Git revisions:

| Skill | Content SHA-256 | Applied method |
| --- | --- | --- |
| `find-skills/SKILL.md` | `c00eeea0e13e74fe4a9d84ba0a8542205a1b736d65f13134fe1a6647eb14976f` | Prefer relevant installed methods; document discovery limits. |
| `clean-code/SKILL.md` | `3c4115e1bc0ead5b023d9cc2c4f79f3a9273bfd363ae5c7eb11cf67f0096f317` | Centralize duplicated download behavior, use explicit errors, remove unused imports, test behavior. |
| `codebase-design/SKILL.md` | `2c20617f87ec8af6a434859f381b2f061a69b530444e74eb39e78bb016a6d1e2` | One small interface hides transport, cache, and filesystem lifecycle. |
| `brainstorming/SKILL.md` | `74edf03ea6d24ef53db48677b93558d14a979bdf052ca3f57ecdca0c66791608` | Bounded existing-flow change; preserve authorized training behavior. |

Clean-code checkpoint at the time above reviewed names, interface ownership,
error propagation, deduplication, resource release, and behavior tests. Removed
13 repeated download/extract/delete implementations and 12 newly unused `os`
imports. No unrelated refactoring or open implementation issue identified.
Independent review is still required; this checkpoint is not a review approval.

## Independent-review repair and runtime evidence

Review repair checkpoint: 2026-10-14 20:43 PDT (2026-10-15 03:43 UTC).
Independent review of `cac8049` identified a P2 compatibility issue: resolving
required file paths before checking cache completeness rejected existing datasets
accessed through symbolic links. Four added cases reproduced the issue before
repair. Required paths are now checked lexically against the loader's dataset
layout, and complete existing linked caches return without Hub access. Actual
write destinations are resolved and checked only for missing files. Existing
linked files remain untouched; incomplete dataset-directory redirects cannot
cause writes outside the dataset directory. A symlinked configured root remains
supported.

After repair, **48 tests passed** in 0.36 seconds. Six added cases cover linked
whole datasets, processed directories, individual tensor files, partial caches
with existing linked files, rejection of missing-file writes through an external
dataset link, and a symlinked configured root. The two changed Python files parse
with Python 3.8 grammar; `git diff --check` passed. Clean-code recheck covered the
separation of read compatibility from write containment; no additional open
implementation issue was found. Final independent re-review remains separate.

A separate real-runtime check passed against `cac8049`, with source hashes and
full results recorded in the portable
[runtime evidence](huggingface-runtime-evidence.json) (2026-10-15
03:42:10–03:42:17 UTC). Local machine paths are replaced with placeholders;
measurements and source hashes are unchanged. It used Python 3.11.5, PyTorch 2.5.1, PyG 2.6.1, and
huggingface_hub 0.36.2:

- An empty dataset root and Hub cache fetched GOODMotif and GOODCBAS anonymously
  from `citrinegui/GOOD`; both original archive SHA-256 hashes matched, and all
  **35 extracted files** were byte-identical to their source archive entries.
- Actual `load()` calls covered both GOODMotif domains, all three shift modes,
  and **26 graph splits**, plus all **three GOODCBAS node shifts**.
- **Nine** real PyG GCN optimization steps had finite losses/gradients and changed
  model parameters.
- Actual cached constructors passed with network and Hub calls blocked; all
  35 processed file sizes and modification times remained unchanged.
- Actual constructors with `generate=True` skipped a download helper configured
  to fail. This checks the download guard, not synthetic regeneration itself.

This runtime check is representative of graph and node training data flows, not
all 13 dataset classes or the full `goodtg` CLI. Original mirror parity for the
remaining datasets is tracked separately by the migration coordinator.


## Final independent implementation review

Recorded: 2026-10-14 20:44 PDT (2026-10-15 03:44 UTC).
Independent reviewer: `dataset_audit`; reviewed exact implementation target
`cec4207974825890942d5f13d17c18271c09ce01`.

The reviewer confirmed the earlier P2 symbolic-link cache compatibility finding
was fixed, independently reran the focused suite (**48 passed in 0.28 seconds**),
and checked `git diff --check` and the clean worktree. Complete linked caches
return before network or lock access, partial caches preserve existing linked
bytes, and new writes still enforce destination containment. **No remaining
blocking implementation findings.** This review does not claim completed uploads
or runtime validation of all 13 dataset classes.

This final documentation-only update makes the runtime evidence portable and
records the review outcome. JSON parsing, evidence counts, repository links,
source commit/hash consistency, and whitespace were checked. No implementation
suite rerun was needed because executable code did not change.

## Complete mirror and immutable download revision

Recorded: 2026-10-14 20:52 PDT (2026-10-15 03:52 UTC).
The migration coordinator completed the public mirror and verified its final
inventory. Downloads now use the immutable dataset snapshot
`29cf40c24604cefaf95f9bd8eb055608ad09da86`, which includes all 13 ZIP archives,
the dataset license, dataset card, and
[public dataset manifest](https://huggingface.co/datasets/citrinegui/GOOD/blob/29cf40c24604cefaf95f9bd8eb055608ad09da86/dataset-manifest.json).

The manifest records **13 archives**, **9,461,334,090 compressed bytes**,
**47,665,296,358 expanded bytes**, **22 domains**, and **192 loader-required
processed tensor files**. Original ZIP CRC validation passed. Every uploaded
archive's Hugging Face LFS SHA-256 and size matched its original source archive,
all domain-required processed files were present, and remote metadata was
accessible anonymously. These whole-mirror integrity checks complement the
representative actual tensor loading and optimization checks above; they do not
claim full training runs for every dataset.

The only executable change after the independently reviewed implementation is
replacing `DATASET_REVISION = 'main'` with the complete snapshot SHA. The focused
suite passed **48 tests in 0.22 seconds** after this change; the downloader passes
Python 3.8 grammar parsing and `git diff --check`. Clean-code recheck found no
new interface, lifecycle, or error-handling change. A final cold-cache check and
review of this pinned revision are coordinated separately before integration.
