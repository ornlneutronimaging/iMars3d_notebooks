from unittest import TestCase
from pathlib import Path
import os

from __code.utilities import folder


class TestFolder(TestCase):

    def setUp(self):
        _file_path = Path(__file__)

        self.existing_folder = _file_path.parent
        self.non_existing_folder = _file_path.parent / "bad_folder" / "bad_inner_folder"

    def test_start_path_exists(self):
        assert os.path.exists(self.existing_folder)

    def test_start_path_does_not_exists(self):
        assert not os.path.exists(self.non_existing_folder)

        first_existing_folder = folder.find_first_real_dir(start_dir=self.non_existing_folder)
        assert first_existing_folder == self.existing_folder
