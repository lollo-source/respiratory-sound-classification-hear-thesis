# Licences and data access

The original thesis-authored software in this repository is released under the MIT License in `LICENSE`. This package also contains sanitised protocol metadata, saved predictions, and numeric reference material. It does not redistribute clinical audio, private identifier mappings, HeAR weights, or OPERA checkpoints.

The repository MIT License does not relicense HF Lung, SPRSound/BioCAS, HeAR, OPERA, pretrained checkpoints, or any other third-party resource. Users are responsible for obtaining datasets and models from authorised sources and complying with their respective licences and access conditions. The literature values under `published_challenge_reference/` are bibliographic facts retained with paper/table provenance; consult the cited publications for reuse terms.

Authoritative access points and the exact layouts used here are documented in [dataset setup](docs/DATASET_SETUP.md) and [model setup](docs/MODEL_SETUP.md). HF_Lung_V1 and SPRSound/BioCAS are publicly downloadable from their upstream source repositories. The HeAR Hugging Face model requires accepting its terms while signed in. OPERA source and checkpoints are public upstream resources. Access status does not override any upstream licence or terms.

No reversible map from release-local record or patient-group IDs to source identifiers is included.
