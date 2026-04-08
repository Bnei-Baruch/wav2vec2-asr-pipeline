from datasets import load_from_disk

DOWNSAMPLE_FACTOR = 320


def check_ctc_lengths(dataset_path: str) -> list[dict]:
    ds = load_from_disk(dataset_path)
    bad = []
    for i, example in enumerate(ds):
        input_len = len(example["input_values"])
        label_len = len([l for l in example["labels"] if l != -100])
        output_len = input_len // DOWNSAMPLE_FACTOR
        if output_len <= label_len:
            bad.append({
                "index": i,
                "input_len": input_len,
                "output_len": output_len,
                "label_len": label_len,
            })
    return bad


def main():
    for split in ("train", "eval"):
        print(f"Checking '{split}'...")
        bad = check_ctc_lengths(f"./{split}")
        if bad:
            print(f"  Found {len(bad)} bad samples (output_len <= label_len):")
            for s in bad[:20]:
                print(f"    idx={s['index']}  input={s['input_len']}  output={s['output_len']}  label={s['label_len']}")
            if len(bad) > 20:
                print(f"    ... and {len(bad) - 20} more")
        else:
            print("  All samples OK")


if __name__ == "__main__":
    main()
