# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Action indices: forward=0  left=1  right=2  backward=3  wait=4
#
# Internal actions (roi_features: [angle, radius]):
#   angle:  left=0  stay=1  right=2
#   radius: shrink=3  stay=4  grow=5

ROI_CONTRACTING = [
    {"action": 4, "internal_action": 4}
] + [  # radius stay — show full ROI
    {"action": 4, "internal_action": 3}
] * 8  # radius shrink 4.5 → 0.5

FOOD_RIPENING_CLOSEUP = [
    {"action": 4, "internal_action": 1},  # unripe
    {"action": 4, "internal_action": 1},  # unripe
    {"action": 4, "internal_action": 1},  # ripe
    {"action": 4, "internal_action": 1},  # ripe
    {"action": 4, "internal_action": 1},  # rotten
    {"action": 4, "internal_action": 1},  # rotten
    {"action": 4, "internal_action": 1},  # despawn + respawn
]

FOOD_RIPENING_WANDER = [
    {"action": 1, "internal_action": 1},  # left
    {"action": 1, "internal_action": 2},
    {"action": 0, "internal_action": 2},  # forward
    {"action": 0, "internal_action": 0},
    {"action": 0, "internal_action": 1},
    {"action": 2, "internal_action": 1},  # right
    {"action": 0, "internal_action": 2},
    {"action": 0, "internal_action": 0},
    {"action": 1, "internal_action": 1},
    {"action": 0, "internal_action": 1},
    {"action": 0, "internal_action": 1},
    {"action": 1, "internal_action": 1},
    {"action": 0, "internal_action": 0},
    {"action": 0, "internal_action": 2},
    {"action": 4, "internal_action": 1},  # wait
    {"action": 0, "internal_action": 1},
    {"action": 4, "internal_action": 1},
]
