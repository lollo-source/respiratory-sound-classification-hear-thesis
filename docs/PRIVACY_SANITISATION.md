# Privacy sanitisation

Source record identifiers were replaced by stable, release-local sequential identifiers after sorting within the HF and BioCAS universes. HF grouping proxies and BioCAS patient groups were independently replaced by sequential group identifiers. Bootstrap rows retain only the zero-based position in the frozen pseudonymous patient ordering. No salt, lookup table, filename, audio path, annotation path, age, sex, acquisition location, or reversible mapping is distributed.

The sanitisation preserves equality joins, membership, grouping equivalence, class labels, predictions, and draw-index semantics. Quick Verification is numerically identical after sanitisation. Full Reproduction uses the public deterministic dataset adapters with explicit local roots; transient source paths and identifiers are never written to public outputs, which contain only release-local pseudonyms.
