""" Unit tests for the events module. """

from pathlib import Path

from orkans import ROOT_DIR
from orkans import events

TEST_EVENT_DIR = ROOT_DIR / "orkans" / "tests" / "_test_events"


class TestFindNewEvents:
    """Units tests for events.find_new_events function."""

    def test_opera_no_event_file(self):
        """
        The events module checks the last processed event,
        to avoid redundant work. Information about the last
        event is in a separate file. If the file does not exist,
        it is assumed that previously no events have been analyzed.
        Checked for OPERA source files.
        """
        wrong_path = Path("")
        dates = events.find_new_events(wrong_path, "opera")
        assert len(dates) == 24

    def test_opera_one_new_event(self):
        # Now the file exists and the first event has been processed.

        event_dir = TEST_EVENT_DIR / "1"
        dates = events.find_new_events(event_dir, "opera")
        assert len(dates) == 23

    def test_opera_no_new_events(self):
        # All events have been processed. No new events.
        event_dir = TEST_EVENT_DIR / "2"
        dates = events.find_new_events(event_dir, "opera")
        assert len(dates) == 0

    def test_opera_some_new_events(self):
        # Some of the events have been processed.
        event_dir = TEST_EVENT_DIR / "3"
        dates = events.find_new_events(event_dir, "opera")
        assert len(dates) == 14

    def test_other_no_new_event(self):
        """
        Different data sources have different file structures.
        The function should be able to deal with this.
        The default fmt (format) is for data from OPERA,
        so for other sources it should be specified.
        """

        wrong_path = Path("")
        dates = events.find_new_events(
            wrong_path,
            "mrms",
            fmt="%Y%m%d-%H%M%S",
        )
        assert len(dates) == 36

    def test_other_new_events(self):
        # Different source, but some events already processed.
        event_dir = TEST_EVENT_DIR / "4"
        dates = events.find_new_events(
            event_dir,
            "mrms",
            fmt="%Y%m%d-%H%M%S",
        )
        assert len(dates) == 25
