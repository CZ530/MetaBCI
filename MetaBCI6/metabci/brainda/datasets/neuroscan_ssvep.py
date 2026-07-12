# -*- coding: utf-8 -*-
"""Local NeuroScan SSVEP dataset adapter.

This wraps the local CNT files under ``数据集2`` in the same style as the
datasets shipped with MetaBCI. Each subject is stored in a folder whose CNT
file has the same stem, for example ``数据集2/LJ/LJ.cnt``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from mne.io import Raw, read_raw_cnt

from .base import BaseDataset
from ..utils.channels import upper_ch_names


class NeuroScanSSVEP(BaseDataset):
    """Local 6-class SSVEP CNT dataset recorded with NeuroScan."""

    _EVENTS = {
        "target_1": (1, (0.14, 2.14)),
        "target_2": (2, (0.14, 2.14)),
        "target_3": (3, (0.14, 2.14)),
        "target_4": (4, (0.14, 2.14)),
        "target_5": (5, (0.14, 2.14)),
        "target_6": (6, (0.14, 2.14)),
    }

    _CHANNELS = [
        "POZ",
        "PZ",
        "PO3",
        "PO5",
        "PO4",
        "PO6",
        "O1",
        "OZ",
        "O2",
    ]

    _DEFAULT_SUBJECTS = ["LJ", "LSW", "LZX", "SY", "ZB", "ZC", "ZCG", "ZH", "ZPC"]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        subjects: Optional[List[str]] = None,
    ):
        project_root = Path(__file__).resolve().parents[4]
        self.data_dir = Path(data_dir) if data_dir is not None else project_root / "数据集2"

        if subjects is None:
            if self.data_dir.exists():
                subjects = sorted(
                    p.name for p in self.data_dir.iterdir()
                    if p.is_dir() and (p / f"{p.name}.cnt").exists()
                )
            else:
                subjects = list(self._DEFAULT_SUBJECTS)

        super().__init__(
            dataset_code="neuroscan_ssvep",
            subjects=subjects,
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm="ssvep",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        subject = str(subject)
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        data_dir = Path(path) if path is not None else self.data_dir
        run_file = data_dir / subject / f"{subject}.cnt"
        if not run_file.exists():
            raise FileNotFoundError(f"NeuroScanSSVEP data file not found: {run_file}")
        return [[run_file]]

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        sessions: Dict[str, Dict[str, Raw]] = {}
        for isess, run_dests in enumerate(dests):
            runs: Dict[str, Raw] = {}
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_cnt(run_file, preload=True, verbose=verbose)
                raw = upper_ch_names(raw)
                runs[f"run_{irun}"] = raw
            sessions[f"session_{isess}"] = runs
        return sessions
