import sys
from file_utils import read_file_lines, write_outputs_to_file, Output

VOCAB_SIZE = 300000


def compute_word_stats(lines: list[str]) -> tuple[dict[str, int], int]:
    word_counts = {}
    total_events = 0
    for line in lines:
        words = line.split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            total_events += 1
    
    return word_counts, total_events


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


lines = read_file_lines(dev_file_name)
dev_dict, num_of_events = compute_word_stats(lines)

outputs.append(Output(7, [str(num_of_events)]))


write_outputs_to_file(output_file_name, outputs)
