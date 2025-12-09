import sys
from collections import Counter
from file_utils import read_file_lines, write_outputs_to_file, Output

VOCAB_SIZE = 300000


def make_dict(words: list[str]) -> dict[str, int]:
    return dict(Counter(words))


def split_to_words(lines: list[str]) -> list[str]:
    words = []
    for line in lines:
        line_words = line.split()
        for word in line_words:
            words.append(word)    
    return words

class Sample:
    def __init__(self, events: list[str]):
        self.events = events
        self.size = len(events)
        self.dict: dict[str, int] = make_dict(events)

    def unique_events_count(self):
        return len(self.dict.keys())
    
    def count_word(self, word: str) -> int:
        return self.dict.get(word, 0)
    
    def mle(self, word: str) -> int:
        return self.dict.get(word, 0) / self.size

# Parse command line arguments
if len(sys.argv) != 5:
    print("Error: Expected 4 arguments")
    print("Usage: python args_parser.py <dev_file_name> <test_file_name> <input_word> <output_file_name>")
    sys.exit(1)

dev_file_name = sys.argv[1]
test_file_name = sys.argv[2]
input_word = sys.argv[3]
output_file_name = sys.argv[4]


outputs: list[Output] = []

outputs.append(Output(1, [dev_file_name]))
outputs.append(Output(2, [test_file_name]))
outputs.append(Output(3, [input_word]))
outputs.append(Output(4, [output_file_name]))
outputs.append(Output(5, [VOCAB_SIZE]))
outputs.append(Output(6, [1/VOCAB_SIZE]))


dev_lines = read_file_lines(dev_file_name)
dev_words = split_to_words(dev_lines)

num_of_90_percent = round(len(dev_words) * 0.9)

training_set = Sample(dev_words[:num_of_90_percent])
validation_set = Sample(dev_words[num_of_90_percent:])

outputs.append(Output(7, [str(len(dev_words))]))
outputs.append(Output(8, [str(validation_set.size)]))
outputs.append(Output(9, [str(training_set.size)]))


outputs.append(Output(10, [str(training_set.unique_events_count())]))
outputs.append(Output(11, [str(training_set.count_word(input_word))]))
outputs.append(Output(12, [str(training_set.mle(input_word))]))
outputs.append(Output(13, [str(training_set.mle('unseen-word'))]))


write_outputs_to_file(output_file_name, outputs)
