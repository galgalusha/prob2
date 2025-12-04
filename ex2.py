import sys

VOCAB_SIZE = 300000

class Output:
    def __init__(self, number: int, lines: list[str]):
        self.number = number
        self.lines = lines


def write_outputs_to_file(file_name: str, outputs: list[Output]):
    with open(file_name, 'w') as f:
        f.write("#Students Gal Koren 040459612 Ekaterina Plechova 345731764\n")
        for output in outputs:
            if len(output.lines) == 1:
                f.write(f"#Output{output.number}\t{output.lines[0]}\n")
            else:
                # Multiple lines: use newline as delimiter
                f.write(f"#Output{output.number}\n")
                for line in output.lines:
                    f.write(f"{line}\n")


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

write_outputs_to_file(output_file_name, outputs)





