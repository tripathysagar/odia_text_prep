import regex as re
from typing import List, Tuple
from functools import lru_cache
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors
from tokenizers.normalizers import NFKD, StripAccents
from tokenizers.pre_tokenizers import UnicodeScripts, Whitespace, ByteLevel

class Odia:
    # List of valid Unicode ranges for Odia script
    valid_ranges = {
        'anusvara': range(0x0B01, 0x0B03 + 1),
        'matras': frozenset(range(0x0B3C, 0x0B4D + 1)).union({0x0B55, 0x0B56, 0x0B57}),
        'digits': range(0x0B66, 0x0B6F + 1),
        'sign': frozenset(range(0x0B72, 0x0B77 + 1)).union(set([0x2018, 0x2019, 0x201C, 0x201D])),
        'aux_sign': frozenset({0x0B70, 0x0964, 0x0965}),
        'vowels': range(0x0B05, 0x0B14 + 1),
        'consonants': frozenset(range(0x0B15, 0x0B39 + 1)).union({0x0B5F, 0x0B71}),
    }

    # List of Unicode code points to ignore
    ignore_case = frozenset({0x0B0D, 0x0B0E, 0x0B11, 0x0B12, 0x0B29, 0x0B31, 0x0B34, 0x0B45, 0x0B46, 0x0B5E, 0x0B49, 0x0B4A})

    def __init__(self):
        self.odia_chars = {
            key: tuple(chr(i) for i in (val if isinstance(val, range) else val) if i not in self.ignore_case)
            for key, val in self.valid_ranges.items()
        }

        self.odia_chars['complex_char'] = tuple(
            ''.join([j, i])
            for i in self.odia_chars['matras'] + self.odia_chars['anusvara']
            for j in self.odia_chars['vowels'] + self.odia_chars['consonants']
        )

    @lru_cache(maxsize=1)
    def generate_pattern(self):
        """
        Generate a regex pattern that matches any valid Odia character.
        patrn  = [
            (r'\s*' + r'|\s*'.join(map(re.escape, chars))) if key == 'complex_char'
            else f"[{''.join(map(re.escape, chars))}]"
            for key, chars in reversed(self.odia_chars.items())
        ]
        return  r'|\s*'.join(patrn)
        """
        all_chars = self.get_all_chars()
        return '\s*' + '|\s*'.join( all_chars)


    @lru_cache(maxsize=1)
    def get_all_chars(self) -> Tuple[str]:
        """
        Get a set of all valid Odia characters. without the complex characters.
        """
        return tuple(j for i in list(reversed(self.odia_chars.keys())) if i != 'complex_char' for j in self.get_chars(i))


    def get_chars(self, category: str) -> List[str]:
        if category not in self.odia_chars:
            raise ValueError(f"Invalid category: {category}")
        return self.odia_chars[category]

    def __getattr__(self, name: str) -> List[str]:
        if name in self.odia_chars:
            return self.get_chars(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def match_odia(self, text):
        """
        Match Odia text using the generated pattern.
        """
        pattern = re.compile(self.generate_pattern())
        return pattern.findall(text)
    
od = Odia()


zero_order = ['vowels', 'consonants', 'digits', 'matras', 'sign', 'aux_sign', 'anusvara' ]
all_chars = [v for k in zero_order for v in od.get_chars(k)]
init_vocab = { k:idx for idx, k in enumerate(all_chars)}

init_merges = []
for idx, st in enumerate(od.get_chars('complex_char'), start=len(init_vocab)):
    init_vocab[st] = idx
    init_merges.append(tuple(st))

init_vocab[" "] = len(init_vocab)
print(f"len of init_vocab : {len(init_vocab)} len of init_merges : {len(init_merges)}")



tokenizer = Tokenizer(models.BPE(vocab=init_vocab,
                                 merges=init_merges,
                                 unk_token="[UNK]",
                                 ))

tokenizer.normalizer = normalizers.Sequence([NFKD(), normalizers.Replace(r'\s+', ' '),])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([ ByteLevel( )])
#tokenizer.post_processor = processors.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
        vocab_size=50_304,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
        show_progress=True,
        min_frequency=2,
        #initial_alphabet=list(od.get_all_chars()),
        )

files = ["wikipedia/odia_wiki_full/train.txt", "wikipedia/odia_wiki_full/valid.txt"]


def file_iter():
    for path in files:
        with open(path, "rt") as f:
            for line in f:
                yield line

tokenizer.train_from_iterator(file_iter(), trainer=trainer)

tokenizer.save("od_tokenizer_hf.json")