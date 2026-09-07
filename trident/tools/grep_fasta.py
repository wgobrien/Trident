import gzip
import os

from taxutils.taxutils import parse_accession


def read_accession_query(value):
    if not isinstance(value, (str, os.PathLike)):
        return [str(token) for token in value if str(token)]
    if os.path.exists(value):
        with open(value) as f:
            text = f.read()
    else:
        text = value
    return [token for token in text.replace(",", " ").split() if token]


def parse_requested_accessions(value, version=True):
    query_values = read_accession_query(value)
    accessions = set()
    for raw_value, parsed_value in zip(query_values, parse_accession(query_values, version=version)):
        raw_value = str(raw_value).strip()
        if raw_value:
            accessions.add(raw_value)
            accessions.add(raw_value.lstrip(">"))
        if parsed_value != "NA":
            accessions.add(parsed_value)
    if not accessions:
        return set()
    return accessions


def _header_match_values(header, parsed_accession):
    text = header.decode("utf-8", errors="ignore").strip()
    name = text[1:].split()[0] if text.startswith(">") else text.split()[0]
    values = {name, name.lstrip(">")}
    if parsed_accession != "NA":
        values.add(parsed_accession)
    return values


def iter_fasta_records(fasta_path, buffer_size):
    header = None
    sequence = []
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    with opener(fasta_path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if header is not None:
                    yield header, sequence
                header = line
                sequence = []
            else:
                sequence.append(line)
        if header is not None:
            yield header, sequence


def write_matching_records(out_f, records, requested_accessions, version=True, verbose=False):
    headers = [header.decode("utf-8", errors="ignore") for header, _ in records]
    accessions = parse_accession(headers, version=version)
    scanned = 0
    matched = 0
    missing_accession = 0
    lines = []

    for (header, sequence), accession in zip(records, accessions):
        scanned += 1
        if accession == "NA":
            missing_accession += 1
            if verbose:
                print(f"NA accession: {header.decode('utf-8', errors='ignore').strip()}")
        if _header_match_values(header, accession).intersection(requested_accessions):
            lines.append(header)
            lines.extend(sequence)
            matched += 1

    if lines:
        out_f.writelines(lines)

    return scanned, matched, missing_accession


def grep_fasta(input_path, accession_query, output_path, version=True, batch_size=1_000_000, verbose=False):
    requested_accessions = parse_requested_accessions(accession_query, version=version)
    if not requested_accessions:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        open(output_path, "wb").close()
        return {
            "requested": 0,
            "scanned": 0,
            "matched": 0,
            "missing_accession": 0,
        }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    totals = {
        "requested": len(requested_accessions),
        "scanned": 0,
        "matched": 0,
        "missing_accession": 0,
    }
    records = []
    buffered_bytes = 0

    with open(output_path, "wb", buffering=batch_size) as out_f:
        for header, sequence in iter_fasta_records(input_path, batch_size):
            records.append((header, sequence))
            buffered_bytes += len(header) + sum(len(line) for line in sequence)
            if buffered_bytes >= batch_size:
                scanned, matched, missing_accession = write_matching_records(
                    out_f,
                    records,
                    requested_accessions,
                    version=version,
                    verbose=verbose,
                )
                totals["scanned"] += scanned
                totals["matched"] += matched
                totals["missing_accession"] += missing_accession
                records = []
                buffered_bytes = 0

        if records:
            scanned, matched, missing_accession = write_matching_records(
                out_f,
                records,
                requested_accessions,
                version=version,
                verbose=verbose,
            )
            totals["scanned"] += scanned
            totals["matched"] += matched
            totals["missing_accession"] += missing_accession

    return totals
