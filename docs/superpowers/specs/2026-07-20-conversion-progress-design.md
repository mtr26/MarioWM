# Conversion Progress Design

## Goal

Make full world-model cache preparation visibly active over SSH and `tmux`,
including both the frame-conversion phase and the SHA-256 hashing phase.

## Behavior

Cache preparation will display two sequential `tqdm` progress bars on stderr:

1. `Converting transitions` advances by the number of source transitions fully
   validated, resized, and written. Its total is the converted transition count.
2. `Hashing cache` advances by bytes read across all generated NumPy artifacts.
   Its total is the sum of their file sizes.

Each bar exposes rate, elapsed time, and ETA through standard `tqdm` output.
Existing conversion output, cache contents, validation, atomic publication, and
Hugging Face upload behavior remain unchanged. Hugging Face continues to own
upload progress reporting.

## Implementation

The conversion loop will update one explicit transition-count progress object
after each HDF5 chunk is completely written. The SHA-256 helper will accept an
optional progress object and update it after each block is read; cache creation
will provide a shared byte-count progress object while cache validation will
continue hashing without an additional bar.

## Verification

An integration test will convert the synthetic HDF5 fixture using real `tqdm`
output captured from stderr and assert that both labeled phases reach 100%.
The complete test suite and a bounded real-data conversion will then run.
