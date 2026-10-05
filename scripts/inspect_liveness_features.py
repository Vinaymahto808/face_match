"""Ad-hoc diagnostic: print the passive-liveness feature separation."""

import sys

sys.path.insert(0, ".")

from app.services.liveness.passive import extract_features, score_features
from tests.conftest import make_face_image, make_spoof_image

CROP = (150, 110, 340, 300)


def crop(img):
    x, y, w, h = CROP
    return img[y : y + h, x : x + w]


live = extract_features(crop(make_face_image(seed=3)))
spoof = extract_features(crop(make_spoof_image(seed=3)))
live_score, live_sub, _ = score_features(live)
spoof_score, spoof_sub, _ = score_features(spoof)

header = "feature".ljust(20) + "live".rjust(11) + "spoof".rjust(11) + "  sub_live sub_spoof"
print(header)
print("-" * len(header))
for key in sorted(live):
    print(
        key.ljust(20)
        + f"{live[key]:11.4f}"
        + f"{spoof[key]:11.4f}"
        + f"{live_sub[key]:9.2f}"
        + f"{spoof_sub[key]:10.2f}"
    )
print()
print(f"SCORE live={live_score:.3f}  spoof={spoof_score:.3f}  gap={live_score - spoof_score:.3f}")
