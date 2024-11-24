import os
# Set environment variables to limit threading
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

from datasets import load_dataset
dataset = load_dataset("OdiaGenAIdata/pre_train_odia_data_processed")

if 'source' in dataset['train'].column_names:
    dataset=dataset.remove_columns('source')

def batch_iterator(batch_size=1000):
    # Iterate over batches of the "text" column in the "train" split
    for batch in dataset["train"].select_columns(["text"]).iter(batch_size=batch_size):
        yield batch["text"]

from tokenizers import Regex, Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors, Regex
from tokenizers.normalizers import NFKD, Replace, NFD, NFKC
from tokenizers.pre_tokenizers import UnicodeScripts, Whitespace, ByteLevel, Split, WhitespaceSplit
from tokenizers.decoders import ByteLevel

tokenizer = Tokenizer(models.BPE(fuse_unk = True, unk_token="[UNK]"))
# \u0964\u0965\u2018\u2019\u201C\u201D -> ।॥‘’“”
# For viram punctuation, use the generic Indic 0964 and 0965. https://www.unicode.org/charts/PDF/U0B00.pdf
odia_and_english_regex = r'[^\u0000-\u00FF\u0B00-\u0B7F\u0964\u0965\u2018\u2019\u201C\u201D]'  # Matches characters outside English and Odia

tokenizer.normalizer = normalizers.Sequence([
    #Replace(r'\u200D', ''),  # Removes Zero Width Joiner
    #Replace(r'\u200C', ''),  # Removes Zero Width Non-Joiner
    Replace(Regex(odia_and_english_regex), ''),  # Removes characters outside English and Odia ranges
    Replace(r'\s+', ' '),
    NFKC(),
])

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    UnicodeScripts(),  # Segment based on script boundaries
    ])

tokenizer.decoder = ByteLevel()

trainer = trainers.BpeTrainer(
        vocab_size=20_000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
        show_progress=True,
        min_frequency=4,
        #initial_alphabet=list(od.get_all_chars()),
        )

tokenizer.train_from_iterator(batch_iterator(50), trainer=trainer,  length=len(dataset['train']))

tokenizer.save('tokenizer_bpe_1.json')
