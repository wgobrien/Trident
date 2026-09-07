import os
import tempfile
from pathlib import Path

from taxutils.taxutils import parse_accession


def write_accession_batch(out_f, headers, line_numbers):
    accessions = parse_accession(headers)
    for accession, header, line_number in zip(accessions, headers, line_numbers):
        if accession == "NA":
            raise ValueError(
                f"No accession found in FASTA header on line {line_number}: "
                f"{header.strip()}"
            )
        out_f.write(f"{accession}\n")


def extract_accessions(fasta_path, output_path, batch_size=10_000):
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    fasta_path = Path(fasta_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    os.close(fd)

    header_count = 0
    headers = []
    line_numbers = []

    try:
        with fasta_path.open() as in_f, open(temporary_path, "w") as out_f:
            for line_number, line in enumerate(in_f, start=1):
                if not line.startswith(">"):
                    continue

                headers.append(line)
                line_numbers.append(line_number)
                if len(headers) >= batch_size:
                    write_accession_batch(out_f, headers, line_numbers)
                    header_count += len(headers)
                    headers.clear()
                    line_numbers.clear()

            if headers:
                write_accession_batch(out_f, headers, line_numbers)
                header_count += len(headers)

        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return header_count
