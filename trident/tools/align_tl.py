import subprocess
import os
import pandas as pd
import numpy as np
import pysam
from collections import defaultdict
from scipy.sparse import csr_matrix
import gzip, shutil
import requests
import time
import re
from dataclasses import dataclass, field

from trident.global_utils import get_sample_folder, get_logger, get_sample_folder
from .grep_fasta import grep_fasta

logger = get_logger(__name__)

@dataclass
class contigs:
    header: str = ""
    sequence: str = ""
    blast_closest_accs: list = field(default_factory=list)
    blast_closest_taxids: list = field(default_factory=list)
    blast_closest_organisms: list = field(default_factory=list)
    blast_evalues: list = field(default_factory=list)
    blast_bitscores: list = field(default_factory=list)
    blast_pct_identity: list = field(default_factory=list)
    blast_query_coverage: list = field(default_factory=list)

    def __repr__(self):
        return f"contig('{self.header}')"
    
def parse_fasta(filepath):
    results = []
    current = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    results.append(current)
                current = contigs()
                current.header = line[1:]
                current.sequence = ""
                current.blast_closest_accs = []
                current.blast_closest_taxids = []
                current.blast_closest_organisms = []
                current.blast_evalues = []
                current.blast_bitscores = []
                current.blast_pct_identity = []
                current.blast_query_coverage = []
            elif current:
                current.sequence += line
        if current:
            results.append(current)
    return results

def blast_sequence(sequence, program="blastn", database="nt", hitlist_size=5):
    submit_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    
    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "HITLIST_SIZE": hitlist_size,
        "FORMAT_TYPE": "XML",
    }
    response = requests.post(submit_url, data=params)
    
    rid, rtoe = None, 15
    for line in response.text.split("\n"):
        if "RID =" in line:
            rid = line.split("=")[1].strip()
        if "RTOE =" in line:
            try:
                rtoe = int(line.split("=")[1].strip())
            except:
                rtoe = 15
    if not rid:
        raise ValueError(f"Could not get RID. Response:\n{response.text[:500]}")
    
    print(f"  RID: {rid}, estimated wait: {rtoe}s")
    time.sleep(rtoe + 5)
    
    for attempt in range(20):
        check_resp = requests.get(submit_url, params={
            "CMD": "Get", "RID": rid,
            "FORMAT_TYPE": "XML", "FORMAT_OBJECT": "SearchInfo"
        })
        status_text = check_resp.text
        if "Status=WAITING" in status_text:
            print(f"  Waiting... (attempt {attempt+1})")
            time.sleep(15)
        elif "Status=FAILED" in status_text:
            raise RuntimeError(f"BLAST job {rid} failed")
        elif "Status=UNKNOWN" in status_text:
            raise RuntimeError(f"BLAST job {rid} expired")
        elif "Status=READY" in status_text:
            break
    else:
        raise RuntimeError("Timed out waiting for BLAST results")
    
    result_resp = requests.get(submit_url, params={
        "CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"
    })
    xml = result_resp.text
    query_len = int(re.search(r"<BlastOutput_query-len>(\d+)</BlastOutput_query-len>", xml).group(1)) if re.search(r"<BlastOutput_query-len>(\d+)</BlastOutput_query-len>", xml) else len(sequence)

    hits = re.findall(r"<Hit>(.*?)</Hit>", xml, re.DOTALL)
    
    accs, taxids, organisms, evalues, bitscores, pct_ids, coverages = [], [], [], [], [], [], []
    
    for hit in hits[:hitlist_size]:
        acc = re.search(r"<Hit_accession>(.*?)</Hit_accession>", hit)
        defn = re.search(r"<Hit_def>(.*?)</Hit_def>", hit)
        
        # Take best HSP (first one)
        evalue = re.search(r"<Hsp_evalue>(.*?)</Hsp_evalue>", hit)
        bitscore = re.search(r"<Hsp_bit-score>(.*?)</Hsp_bit-score>", hit)
        identity = re.search(r"<Hsp_identity>(\d+)</Hsp_identity>", hit)
        align_len = re.search(r"<Hsp_align-len>(\d+)</Hsp_align-len>", hit)
        qfrom = re.search(r"<Hsp_query-from>(\d+)</Hsp_query-from>", hit)
        qto = re.search(r"<Hsp_query-to>(\d+)</Hsp_query-to>", hit)
        
        acc_val = acc.group(1) if acc else ""
        defn_val = defn.group(1).split(">")[0].strip() if defn else ""  # take first organism if multiple
        evalue_val = evalue.group(1) if evalue else ""
        bitscore_val = f"{float(bitscore.group(1)):.1f}" if bitscore else ""
        
        pct_id_val = ""
        if identity and align_len:
            pct_id_val = f"{100 * int(identity.group(1)) / int(align_len.group(1)):.1f}"
        
        coverage_val = ""
        if qfrom and qto and query_len:
            coverage_val = f"{100 * (int(qto.group(1)) - int(qfrom.group(1)) + 1) / query_len:.1f}"
        
        accs.append(acc_val)
        organisms.append(defn_val)
        evalues.append(evalue_val)
        bitscores.append(bitscore_val)
        pct_ids.append(pct_id_val)
        coverages.append(coverage_val)
        taxids.append("")  # filled below
    
    # Fetch taxids via eutils for all accs at once
    if accs:
        taxids = fetch_taxids_for_accs(accs)
    
    return accs, taxids, organisms, evalues, bitscores, pct_ids, coverages


def fetch_taxids_for_accs(accs):
    ids = ",".join(accs)
    resp = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params={
        "db": "nuccore",
        "id": ids,
        "rettype": "docsum",
        "retmode": "json"
    })
    try:
        data = resp.json()
        return [str(data["result"][uid].get("taxid", "")) for uid in data["result"]["uids"]]
    except Exception:
        return [""] * len(accs)


def build_header(c):
    """Reconstruct FASTA header with BLAST annotations appended."""
    # Strip any previous blast annotations we may have added
    base = c.header.split(" blast_top_acc=")[0]
    
    if not c.blast_closest_accs:
        return base
    
    top_acc = c.blast_closest_accs[0]
    top_taxid = c.blast_closest_taxids[0] if c.blast_closest_taxids else ""
    top_org = c.blast_closest_organisms[0] if c.blast_closest_organisms else ""
    top_evalue = c.blast_evalues[0] if c.blast_evalues else ""
    top_bitscore = c.blast_bitscores[0] if c.blast_bitscores else ""
    top_pct_id = c.blast_pct_identity[0] if c.blast_pct_identity else ""
    top_cov = c.blast_query_coverage[0] if c.blast_query_coverage else ""
    
    # Clean organism name for header safety
    top_org_clean = top_org.replace(" ", "_").replace(",", "").replace(";", "")
    
    annotation = (
        f" blast_top_acc={top_acc}"
        f" blast_top_taxid={top_taxid}"
        f" blast_top_org={top_org_clean}"
        f" blast_evalue={top_evalue}"
        f" blast_bitscore={top_bitscore}"
        f" blast_pct_id={top_pct_id}"
        f" blast_qcov={top_cov}"
    )
    return base + annotation


def _has_blast_annotation(header):
    """Return whether a FASTA header contains a completed BLAST annotation."""
    return re.search(r"(?:^|\s)blast_top_acc=\S+", header) is not None


def blast_contigs(acc_adata):
    filepath = acc_adata.uns["trident"]["contigs"]
    contig_list = parse_fasta(filepath)
    
    tmp_path = filepath + ".blast_tmp"
    
    try:
        with open(tmp_path, "w") as tmp_f:
            for c in contig_list:
                if _has_blast_annotation(c.header):
                    print(f"Skipping {c.header.split(' blast_top_acc=')[0]} (already annotated)...")
                    tmp_f.write(f">{c.header}\n")
                    for i in range(0, len(c.sequence), 60):
                        tmp_f.write(c.sequence[i:i+60] + "\n")
                    tmp_f.flush()
                    continue

                print(f"BLASTing {c.header}...")
                try:
                    (c.blast_closest_accs,
                     c.blast_closest_taxids,
                     c.blast_closest_organisms,
                     c.blast_evalues,
                     c.blast_bitscores,
                     c.blast_pct_identity,
                     c.blast_query_coverage) = blast_sequence(c.sequence)
                    
                    print(f"  Top hit: {c.blast_closest_accs[0]} | {c.blast_closest_organisms[0]} | "
                          f"e={c.blast_evalues[0]} | {c.blast_pct_identity[0]}% id | {c.blast_query_coverage[0]}% cov")
                except Exception as e:
                    print(f"  Failed: {e}")
                
                new_header = build_header(c)  # returns base header if blast failed
                tmp_f.write(f">{new_header}\n")
                for i in range(0, len(c.sequence), 60):
                    tmp_f.write(c.sequence[i:i+60] + "\n")
                tmp_f.flush()
        
        os.replace(tmp_path, filepath)
    
    except Exception as e:
        print(f"  Fatal error: {e}. Original file left untouched.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    
    return contig_list

    
def cluster_consensus(acc_adata, output_path=None):
    """Build consensus sequences per read cluster using de novo assembly."""
    base_path = output_path
    if base_path is None:
        base_path = get_sample_folder(acc_adata.uns["trident"]["basename"])
        
    config = acc_adata.uns["trident"]["config"]
    threads = config.get("threads", 1)
    reads_path = config["reads"]
    pair_path = config.get("pair", None)
    
    clusters = acc_adata.obs["read_cluster"].unique()
    consensuses = {}
    
    for cluster in sorted(clusters):
        logger.info(f"Processing cluster {cluster}")
        
        mask = acc_adata.obs["read_cluster"] == cluster
        read_ids = set(acc_adata.obs.loc[mask, "read_id"].tolist())
        logger.info(f"  {len(read_ids)} reads")
        
        cluster_fq = os.path.join(base_path, f"cluster_{cluster}.fastq")
        _extract_reads(reads_path, read_ids, cluster_fq, pair_path=pair_path)
        
        asm_dir = os.path.join(base_path, f"cluster_{cluster}_asm")
        if os.path.exists(asm_dir):
            shutil.rmtree(asm_dir)
        
        subprocess.run([
            "megahit",
            "-r", cluster_fq,
            "-o", asm_dir,
            "--min-contig-len", "200",
            "-t", str(threads),
        ], check=True)
        
        contigs_path = os.path.join(asm_dir, "final.contigs.fa")
        
        contigs = {}
        with open(contigs_path) as f:
            name, seq = None, []
            for line in f:
                if line.startswith('>'):
                    if name:
                        contigs[name] = ''.join(seq)
                    name = line.strip().lstrip('>')
                    seq = []
                else:
                    seq.append(line.strip())
            if name:
                contigs[name] = ''.join(seq)
        
        consensuses[cluster] = contigs
        os.remove(cluster_fq)
        shutil.rmtree(asm_dir)
        if len(contigs.values()) == 0:
            continue
        else:
            logger.info(f"  {len(contigs)} contigs, longest: {max(len(s) for s in contigs.values())} bp")
        
    output_fasta = os.path.join(base_path, "cluster_contigs.fasta")
    with open(output_fasta, 'w') as f:
        for cluster, contigs in consensuses.items():
            for name, seq in contigs.items():
                f.write(f">cluster_{cluster}_{name}\n{seq}\n")
    
    logger.info(f"Wrote contigs to `.uns[\"trident\"][\"contigs\"]`")
    acc_adata.uns["trident"]["contigs"] = output_fasta

def _extract_reads(reads_path, read_ids, output_path, pair_path=None):
    """Extract reads by ID from fastq."""
    
    opener = gzip.open if reads_path.endswith('.gz') else open
    
    with opener(reads_path, 'rt') as fin, open(output_path, 'w') as fout:
        write = False
        for i, line in enumerate(fin):
            if i % 4 == 0:
                rid = line.strip().split()[0].lstrip('@')
                write = rid in read_ids
            if write:
                fout.write(line)
    
    if pair_path:
        with opener(pair_path, 'rt') as fin, open(output_path, 'a') as fout:
            write = False
            for i, line in enumerate(fin):
                if i % 4 == 0:
                    rid = line.strip().split()[0].lstrip('@')
                    write = rid in read_ids
                if write:
                    fout.write(line)

def compare_references(acc_adata, config):
    """Compute all-vs-all pairwise identity of references."""
    base_path = get_sample_folder(acc_adata.uns["trident"]["basename"])
    
    accessions = acc_adata.var.accession.tolist()
    logger.info("Comparing %d selected accession references.", len(accessions))
    
    # Full accession list for matrix indexing
    all_accessions = acc_adata.var["accession"].tolist()
    acc_to_idx = {a: i for i, a in enumerate(all_accessions)}
    T = len(all_accessions)
    
    fasta_path = os.path.join(base_path, "selected_accessions.fasta")
    paf_path = os.path.join(base_path, "self_alignment.paf")
    
    threads = config.threads
    totals = grep_fasta(config.library, accessions, fasta_path)
    if totals["matched"] == 0:
        logger.warning(
            "No selected accessions were found in %s. Skipping reference similarity adjustment.",
            config.library,
        )
        return csr_matrix((T, T))
    if totals["matched"] < totals["requested"]:
        logger.warning(
            "Only selected %d/%d requested references from %s.",
            totals["matched"],
            totals["requested"],
            config.library,
        )
    logger.info(
        "Selected %d/%d references from %s.",
        totals["matched"],
        totals["requested"],
        config.library,
    )

    minimap_cmd = [
        "minimap2",
        "-x", "asm20",
        "-t", str(threads),
        "-c",
        "--secondary=yes",
        "-N", "10000",
        "-p", "0.4",
        fasta_path, fasta_path,
        "-o", paf_path
    ]
    subprocess.run(minimap_cmd, check=True)
    
    # Aggregate across all alignment blocks per pair
    pair_data = defaultdict(lambda: {'matches': 0, 'query_len': 0, 'target_len': 0})
    
    with open(paf_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            query, target = fields[0], fields[5]
            if query == target:
                continue
            
            pair = tuple(sorted([query, target]))
            direction = (fields[0], fields[5])
            if direction[0] != pair[0]:
                continue
            
            pair_data[pair]['matches'] += int(fields[9])
            pair_data[pair]['query_len'] = int(fields[1])
            pair_data[pair]['target_len'] = int(fields[6])
    
    rows, cols, vals = [], [], []
    for pair, d in pair_data.items():
        i, j = acc_to_idx.get(pair[0]), acc_to_idx.get(pair[1])
        if i is None or j is None:
            continue
        shorter_len = min(d['query_len'], d['target_len'])
        matched = min(d['matches'], shorter_len)
        gi = matched / shorter_len if shorter_len > 0 else 0
        if gi > 0:
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([gi, gi])
    
    mat = csr_matrix((vals, (rows, cols)), shape=(T, T))
    
    os.remove(fasta_path)
    os.remove(paf_path)
    
    return mat

def align(acc_adata, top_n=None, output_path=None):
    base_path = get_sample_folder(acc_adata.uns["trident"]["basename"])
    config = acc_adata.uns["trident"]["config"]

    if top_n is None:
        top_n = acc_adata.var.shape[0]
    
    accessions = acc_adata.var.sort_values(
        "marginal_proba", ascending=False
    ).head(top_n).accession.tolist()
    
    fasta_path = os.path.join(base_path, "selected_accessions.fasta")
    if output_path is None:
        sam_path = os.path.join(base_path, "reads_to_accessions.sam")
    else:
        sam_path = os.path.join(output_path, "reads_to_accessions.sam")
        
    threads = acc_adata.uns['trident']['config']['threads']
    totals = grep_fasta(config["library"], accessions, fasta_path)
    if totals["matched"] == 0:
        logger.warning(
            "No selected accessions were found in %s. Skipping read alignment.",
            config["library"],
        )
        return None
    if totals["matched"] < totals["requested"]:
        logger.warning(
            "Only selected %d/%d requested references from %s.",
            totals["matched"],
            totals["requested"],
            config["library"],
        )
    logger.info(
        "Selected %d/%d references from %s.",
        totals["matched"],
        totals["requested"],
        config["library"],
    )

    minimap_cmd = [
        "minimap2", "-ax", "sr",
        "-t", str(config.get("threads", 1)),
        "--secondary=yes", "-N", "2",
        fasta_path,
        config["reads"]
    ]
    
    if config.get("pair") is not None:
        minimap_cmd.append(config["pair"])
    
    minimap_cmd.extend(["-o", sam_path])
    
    subprocess.run(minimap_cmd, check=True)
    os.remove(fasta_path)

    logger.info("Added alignment filepath to .uns['trident']['alignment']")
    acc_adata.uns["trident"]["alignment"] = sam_path
    return sam_path
