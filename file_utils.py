def read_file_lines(file_name: str) -> list[str]:
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('<TRAIN') and not line.startswith('<TEST') and line:
                lines.append(line)
    return lines


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
