# -*- coding: utf-8 -*-
"""Local NanoEEG SSVEP dataset adapter.

This wraps the local BDF files under ``数据集1`` in the same style as the
datasets shipped with MetaBCI. The files are expected to be named
``1.bdf`` ... ``20.bdf`` and to contain a ``Trigger/Status`` stim channel.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from mne.io import Raw, read_raw_bdf

from .base import BaseDataset
from ..utils.channels import upper_ch_names


class NanoSSVEP(BaseDataset):
    """Local 7-class SSVEP BDF dataset recorded with NanoEEG."""

    _EVENTS = {
        "target_1": (1, (0.14, 2.14)),
        "target_2": (2, (0.14, 2.14)),
        "target_3": (3, (0.14, 2.14)),
        "target_4": (4, (0.14, 2.14)),
        "target_5": (5, (0.14, 2.14)),
        "target_6": (6, (0.14, 2.14)),
        "target_7": (7, (0.14, 2.14)),
    }

    _CHANNELS = [
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "C5",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "P3",
        "PZ",
        "P4",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "O1",
        "OZ",
        "O2",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "CP5",
        "CP3",
    ]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        subjects: Optional[List[int]] = None,
    ):
        project_root = Path(__file__).resolve().parents[4]
        self.data_dir = Path(data_dir) if data_dir is not None else project_root / "数据集1"
        subjects = subjects if subjects is not None else list(range(1, 21))
        super().__init__(
            dataset_code="nano_ssvep",
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
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        data_dir = Path(path) if path is not None else self.data_dir
        run_file = data_dir / f"{subject}.bdf"
        if not run_file.exists():
            raise FileNotFoundError(f"NanoSSVEP data file not found: {run_file}")
        return [[run_file]]

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        sessions: Dict[str, Dict[str, Raw]] = {}
        for isess, run_dests in enumerate(dests):
            runs: Dict[str, Raw] = {}
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_bdf(
                    run_file,
                    preload=True,
                    stim_channel="Trigger/Status",
                    verbose=verbose,
                )
                raw = upper_ch_names(raw)
                runs[f"run_{irun}"] = raw
            sessions[f"session_{isess}"] = runs
        return sessions
