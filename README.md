# Introduction

This script recursively goes through all the PDF files in a given directory (and subdirectories), extracts the title from the PDF metadata. 
If the title is missing, or if it's not good enough, it uses local LLMs to extract the title. 
If the title is still not good enough, it uses another local LLM to try again. 
Then it renames the file to "GoodTitle (original-filename).pdf" or "Unknown-<random> (original-filename).pdf".  
The processing is logged into a SQLite database "processed_files.db". It is useful as the processing of files may be interrupted, and using the database we can resume processing from where we left off.  This script does not modify the database if the file has already been processed.

## Installation

Preparation of python virtual environment:

```bash
python3 -m venv doc-title-finder-env
source doc-title-finder-env/bin/activate
pip install -r requirements.txt
```

## Download local LLMs

The models are automatically downloaded using a helper script `download-llms.sh`. 
Note, that you need to accept the license agreement for the models on Hugging Face before running this script. 

```bash
bash download-llms.sh
```

## Running

In the example, we process documents from the `doc-input` directory. The resulting files are copied to the `doc-output` directory.
The `doc-input` directory contains sample articles from [arXiv](https://arxiv.org/). The filenames of these documents are not meaningful, they represent some identifier of the document, for instance "2412.15754v1". So, the script will try to rename those files based on the title extracted from the document.

Smaller LLMs like MiniCPM-1B work best on consumer hardware. They are faster and require less memory.  But they are prone to generating nonsense. Here in the example, we use it as the first LLM to give us a chance to fix errors for free. If the title is not good enough, we use the second LLM to give us a second chance.

```bash
mkdir doc-output
python doc-title-finder.py models/LFM2.5-350M-Q8_0.gguf models/MiniCPM5-1B-Q4_K_M.gguf doc-input/ doc-output/
```

To process 32 files (on a DELL Laptop equipped with 32GB of RAM and Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz), it takes about 1 minute. But it is worth noting, that not all files have were processed by LLM, those that have meaningful titles in their metadata are left as-is. 

The processing of one document by the LFM2.5-350M-Q8_0 takes about 6-10 seconds. The MiniCPM5-1B-Q4_K_M is only used if the title is not good enough, and it takes about 14-20 seconds to process one document. 

Larger LLMs work best on high-end hardware with dedicated GPUs. They are slower but produce better results. Here is an example of running the script with larger LLMs:

```bash
python doc-title-finder.py models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf models/Qwen3.5-9B.Q8_0.gguf doc-input/ doc-output/
```

During the experiments, one may need to run the script on the same set of input files over and over again. In that case, one can remove the output directory and the processed files database using the following command:

```bash
bash delete-output.sh
```

