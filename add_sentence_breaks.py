def insert_sentence_breaks(infile, outfile):
    with open(infile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(outfile, "w", encoding="utf-8") as out:
        o_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            out.write(line)
            token, tag = stripped.split()
            if tag == "O":
                o_count += 1
            else:
                o_count = 0
            if o_count >= 5:
                out.write("\n")
                o_count = 0

        out.write("\n")

if __name__ == "__main__":
    insert_sentence_breaks("ner_dataset.txt", "ner_dataset_final.txt")
