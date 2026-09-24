# Dataset setup for Full Reproduction

Clinical audio is not included in this repository. Both required datasets are publicly accessible from their upstream repositories; no application or manual access request is documented by those sources. Review and follow each upstream licence and disclaimer. Full Reproduction performs no dataset download and no parent-directory discovery: the two paths supplied on the command line must already contain the layouts below.

## HF Lung

The required dataset is **HF_Lung_V1**, published by Heroic-Faith Medical Science Co. Ltd. The authoritative project is [techsupportHF/HF_Lung_V1 on GitLab](https://gitlab.com/techsupportHF/HF_Lung_V1). The validated source revision was `2a77d37230b1673d332645e6c6afeea29900bcf9`.

Download all multipart train and test archives, then extract the first part of each set with a 7-Zip-compatible program. For example:

```bash
git clone https://gitlab.com/techsupportHF/HF_Lung_V1.git /path/to/HF_Lung_source
git -C /path/to/HF_Lung_source checkout 2a77d37230b1673d332645e6c6afeea29900bcf9
mkdir -p /path/to/HF_Lung
7z x /path/to/HF_Lung_source/train.7z.001 -o/path/to/HF_Lung
7z x /path/to/HF_Lung_source/test.7z.001 -o/path/to/HF_Lung
```

`--hf-lung-root` points to `/path/to/HF_Lung` below, not to the directory containing the multipart archives:

```text
HF_Lung/
├── train/                  # 7,809 WAV files, recursive
└── test/                   # 1,956 WAV files, recursive
```

Sanity check from a POSIX shell:

```bash
find /path/to/HF_Lung/train -type f -iname '*.wav' | wc -l
find /path/to/HF_Lung/test -type f -iname '*.wav' | wc -l
```

The expected total is 9,765 WAV records. Labels and grouping information needed by this release are supplied through its pseudonymised protocol manifest; the adapter deterministically matches that manifest to the authorised local inventory.

## SPRSound / BioCAS 2022 and 2023

The required dataset is **SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database**, specifically the BioCAS 2022 training/test release and BioCAS 2023 test release. The authoritative public source is [SJTU-YONGFU-RESEARCH-GRP/SPRSound on GitHub](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound). The validated source revision was `874eeb8736ddb78937c2fb5332fc7e7293d0f0ca`.

```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git /path/to/SPRSound
git -C /path/to/SPRSound checkout 874eeb8736ddb78937c2fb5332fc7e7293d0f0ca
```

`--sprsound-root` points to `/path/to/SPRSound` below. Full Reproduction consumes the WAV directories; annotation JSON and later challenge years are not runtime inputs.

```text
SPRSound/
├── BioCAS2022/
│   ├── train2022_wav/      # 1,949 raw WAV files
│   └── test2022_wav/       # 734 raw WAV files
└── BioCAS2023/
    └── test2023_wav/       # 871 raw WAV files
```

Sanity check:

```bash
find /path/to/SPRSound/BioCAS2022/train2022_wav -type f -iname '*.wav' | wc -l
find /path/to/SPRSound/BioCAS2022/test2022_wav -type f -iname '*.wav' | wc -l
find /path/to/SPRSound/BioCAS2023/test2023_wav -type f -iname '*.wav' | wc -l
```

The raw inventory must total 3,554 records. The official Challenge workflow retains all of them and derives 1,949 train-2022, 355 test-inter, 379 test-intra and 871 test-2023 records. The harmonised HeAR and OPERA workflows apply the frozen inclusion protocol and use 3,323 records: 1,771 train-2022, 724 test-2022 and 828 test-2023. OPERA uses this same 3,323-record harmonised formulation.

## Adapter and privacy behavior

The adapters validate directory layout, counts, filename uniqueness, split membership and the complete pseudonymous record universe before processing audio. Original filenames and patient identifiers are used only transiently to locate authorised local audio. Generated public manifests and result bundles contain release-local record and group identifiers only.
