"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""

import torch


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_eegs = []
    for eeg, events in batch:
        batch_eegs.append(eeg)
        batch_events.append(events)
    return torch.stack(batch_eegs, 0), batch_events
