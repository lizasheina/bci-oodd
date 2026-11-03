from pathlib import Path
import moabb.datasets as datasets


HANDS = ("left_hand", "right_hand")

import mne
import pooch
import requests
from moabb.datasets.base import BaseDataset

TTK_ROOT = Path.home() / "mne_data" / "TTK"

# Участники A, C, D, E, F, G: первые два дня каждого.
TTK_DAYS = {
    1: (1, 10), 3: (3, 11), 4: (4, 6),
    5: (5, 7), 6: (9, 12), 7: (15, 16),
}
TTK_EVENTS = {
    "right_hand": 5, "left_hand": 7,
    "right_foot": 9, "left_foot": 11,
}


class TTK(BaseDataset):
    def __init__(self, sessions=None):
        super().__init__(
            subjects=list(TTK_DAYS),
            sessions_per_subject=2,
            events=TTK_EVENTS,
            code="TTK",
            interval=[0, 4],
            paradigm="imagery",
            selected_sessions=sessions,
        )

    def data_path(self, subject, path=None, **kwargs):
        if not hasattr(self, "_files"):
            url = (
                "https://repo.researchdata.hu/api/datasets/:persistentId/"
                "?persistentId=hdl:21.15109/CONCORDA/UOQQVK"
            )
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            self._files = [
                f["dataFile"]
                for f in response.json()["data"]["latestVersion"]["files"]
            ]

        prefixes = tuple(f"S{day:03d}R" for day in TTK_DAYS[subject])
        files = sorted(
            (
                f for f in self._files
                if f["filename"].startswith(prefixes)
                and f["filename"].endswith("_raw.fif")
            ),
            key=lambda f: f["filename"],
        )
        if {int(f["filename"][1:4]) for f in files} != set(TTK_DAYS[subject]):
            raise RuntimeError("В релизе TTK не найдены оба выбранных дня.")

        return [
            pooch.retrieve(
                url=f'https://repo.researchdata.hu/api/access/datafile/{f["id"]}',
                known_hash="md5:" + f["checksum"]["value"],
                path=Path(path) if path else TTK_ROOT,
                fname=f["filename"],
                progressbar=True,
            )
            for f in files
        ]

    def _get_single_subject_data(self, subject):
        sessions = {"0": {}, "1": {}}
        names = {code: name for name, code in TTK_EVENTS.items()}

        for path in self.data_path(subject):
            day = int(Path(path).name[1:4])
            session = str(TTK_DAYS[subject].index(day))
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

            raw.annotations.rename({
                desc: names[int(desc.split()[-1])]
                for desc in set(raw.annotations.description)
                if desc.startswith("Stimulus/S")
                and int(desc.split()[-1]) in names
            })
            sessions[session][str(len(sessions[session]))] = raw

        return sessions


datasets.TTK = TTK