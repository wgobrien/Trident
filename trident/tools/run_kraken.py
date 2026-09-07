#!/usr/bin/python

import os
import sys
import argparse
from pathlib import Path
import glob
import shutil

from trident.global_utils import (
    TRIDENT_GLOBALS,
    get_sample_folder,
    get_basename_from_reads,
    get_database_name,
    get_logger
)

logger = get_logger(__name__)

class KrakenPipeline:
    def __init__(self, reads, database, pair=None, filter_reads=False, confidence=0, threads=32, kraken_outputs=True):
        self.reads = reads
        self.pair = pair
        self.database = Path(database)
        self.filter_reads = filter_reads
        self.basename = get_basename_from_reads(reads)

        self.output_folder = Path(get_sample_folder(self.basename))
        self.setup_output_structure()
        self.kraken_file = None if self.output_dne() is True else self.get_output_path()
        
        self.threads = threads
        self.kraken_confidence = confidence
        self.quality_score = 7
        self.kraken_outputs = kraken_outputs

    def get_output_path(self):
        base_name = self.get_base_filename(self.reads)
        base_name += "." + get_database_name(self.database)
        if self.pair is not None:
            base_name += ".paired"
        base_name += ".kraken"
        return os.path.join(self.output_folder, base_name)
        
    def output_dne(self):
        return not os.path.exists(self.get_output_path())

    def setup_output_structure(self):
        """Create organized output directory structure"""
        self.dirs = {
            'filtered_fastq': self.output_folder / 'filtered_fastq',
            'fastp_reports': self.output_folder / 'fastp_reports',
            'kraken': self.output_folder,
        }
        self.dirs['kraken'].mkdir(parents=True, exist_ok=True)

    def check_database_path(self):
        """Get the full path to the specified database"""

        db_path = self.database        
        required_files = {'hash.k2d', 'opts.k2d', 'taxo.k2d'}
        db_files = {f.name for f in db_path.glob('*.k2d')}
        if not required_files.issubset(db_files):
            raise ValueError(f"Invalid Kraken2 database at {db_path}")
            
        return db_path

    def syscommand(self, command):
        """Execute a system command and print it"""
        logger.info(f"Executing: {command}")
        return_code = os.system(command)
        if return_code != 0:
            logger.warning(f"Warning: Command returned non-zero exit code: {return_code}")
        return return_code

    def get_base_filename(self, filepath):
        """Extract base filename without extensions and paths"""
        return Path(filepath).stem.replace('.fastq', '')

    def run_kraken(self, reads, pair=None):
        """Run Kraken2 on input files"""
        db_path = self.check_database_path()
        if not db_path:
            raise ValueError("No database specified")
        
        base_name = self.get_base_filename(reads)
        base_name += "." + get_database_name(self.database)
        if pair is not None:
            base_name += ".paired"
        
        base_output = self.dirs['kraken'] / base_name
        
        cmd = [
            "kraken2",
            f"--threads {self.threads}",
            f"--db {db_path}",
            f"--report {base_output}.kreport",
            f"--output {base_output}.kraken",
            f"--confidence {self.kraken_confidence}",
            f" --memory-mapping"
        ]

        if pair is not None:
            cmd.append(f"{reads} --paired {pair}")
        else:
            cmd.append(reads)

        try:
            self.syscommand(" ".join(cmd))
        except Exception as e:
            logger.error(f"Kraken failed: {e}")
            return base_output
        if not self.kraken_outputs:
            kraken_file = f"{base_output}.kraken"
            if os.path.exists(kraken_file):
                os.remove(kraken_file)
        
        self.kraken_file = f"{base_output}.kraken"
        
        return base_output

    def filtered_fastqs_exist(self, sample_name):
           files = list(self.dirs['filtered_fastq'].glob(f"{sample_name}*.filtered.fastq.gz"))
           expected_count = 2 if self.pair is not None else 1
           return len(files) >= expected_count

    def run_fastp(self, reads, pair=None):
        """Run FastP quality filtering"""
        self.dirs['filtered_fastq'].mkdir(parents=True, exist_ok=True)
        self.dirs['fastp_reports'].mkdir(parents=True, exist_ok=True)
        if pair is not None:
            base_name1 = self.get_base_filename(reads)
            base_name2 = self.get_base_filename(pair)
            
            out1 = self.dirs['filtered_fastq'] / f"{base_name1}.filtered.fastq.gz"
            out2 = self.dirs['filtered_fastq'] / f"{base_name2}.filtered.fastq.gz"
             
            report_html = self.dirs['fastp_reports'] / f"{base_name1}.paired.html"
            report_json = self.dirs['fastp_reports'] / f"{base_name1}.paired.json"
            
            cmd = [
                "fastp",
                f"-i {reads}",
                f"-I {pair}",
                f"-o {out1}",
                f"-O {out2}",
                f"-q {self.quality_score}",
                f"-h {report_html}",
                f"-j {report_json}"
                ]
            
            
            if not self.filtered_fastqs_exist(os.path.basename(reads)):
                self.syscommand(" ".join(cmd))
            else:
                logger.info(f"Skipping fastp for {base_name1}, filtered output already exists.")
            return [str(out1), str(out2)]
        else:
            base_name = self.get_base_filename(reads)
            
            out = self.dirs['filtered_fastq'] / f"{base_name}.filtered.fastq.gz"
            
            report_html = self.dirs['fastp_reports'] / f"{base_name}.html"
            report_json = self.dirs['fastp_reports'] / f"{base_name}.json"
            
            cmd = [
                "fastp",
                f"-i {reads}",
                f"-o {out}",
                f"-q {self.quality_score}",
                f"-h {report_html}",
                f"-j {report_json}"
            ]
            
            if not self.filtered_fastqs_exist(os.path.basename(reads)):
                self.syscommand(" ".join(cmd))
            else:
                logger.info(f"Skipping fastp for {base_name}, filtered output already exists.")
            return [str(out)]

    def run_pipeline(self):
        """Main processing pipeline"""
        current_files = [self.reads, self.pair]
        if self.filter_reads:
            logger.info(f"\nRunning quality filtering on: {', '.join(filter(None, current_files))}")
            current_files = self.run_fastp(reads=self.reads, pair=self.pair)
            logger.info(f"Filtered files saved to: {', '.join(current_files)}")

        logger.info(f"\nRunning Kraken2 with database: {self.database}")
        self.run_kraken(reads=current_files[0], pair=current_files[1] if len(current_files) > 1 else None)
        self.cleanup()

    def cleanup(self):
        for key in ("filtered_fastq", "fastp_reports"):
            dir_path = self.dirs.get(key)
            if dir_path and dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {dir_path}: {e}")
