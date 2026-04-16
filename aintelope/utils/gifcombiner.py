# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from PIL import Image

# --- Define your GIFs here ---
basepath = "/home/joel/project/attention-schema-theory-experiments/outputs/5x5_5trials_dqn_fc/test/"
input_paths = [basepath + "playback" + str(i) + ".gif" for i in range(1, 10)]
output_path = "combined.gif"
# -----------------------------

frames = []
durations = []

for path in input_paths:
    gif = Image.open(path)
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(gif.convert("RGBA"))
        durations.append(gif.info.get("duration", 700))

frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
)

print(f"Saved {len(frames)} frames to {output_path}")
