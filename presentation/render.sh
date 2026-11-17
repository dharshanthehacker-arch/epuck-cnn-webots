#!/bin/zsh
# Build the deck, validate it and render contact sheets with LibreOffice.
# Usage: zsh render.sh <output_dir>
set -e
HERE="${0:A:h}"
SK="/Users/bobby/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/985518ea-9452-4457-9457-86be3a3c7d3f/3ae893ce-94ef-4601-b2b7-7e1c615bb529/skills/pptx"
R="$1"
export PATH=/opt/homebrew/bin:$PATH
mkdir -p "$R"
/Users/bobby/.pyenv/versions/3.9.13/bin/python3 "$HERE/build_deck.py"
cp "$HERE/CNN_Webots_Presentation.pptx" "$R/deck.pptx"
(cd "$SK" && /tmp/v312/bin/python scripts/office/validate.py "$R/deck.pptx" | tail -2)
(cd "$R" && timeout 240 /tmp/v312/bin/python "$SK/scripts/office/soffice.py" --headless --convert-to pdf deck.pptx >/dev/null 2>&1 && pdftoppm -jpeg -r 50 deck.pdf slide && pdftoppm -jpeg -r 110 deck.pdf big)
/tmp/v312/bin/python - "$R" <<'EOF'
import glob, sys
from PIL import Image
R = sys.argv[1]
fs = sorted(glob.glob(f"{R}/slide-*.jpg")); ims = [Image.open(f) for f in fs]; w, h = ims[0].size
print("slides:", len(fs))
for k in range(0, len(ims), 6):
    sh = Image.new("RGB", (w * 2 + 10, (h + 10) * 3), "gray")
    for j, im in enumerate(ims[k:k + 6]):
        sh.paste(im, ((j % 2) * (w + 10), (j // 2) * (h + 10)))
    sh.save(f"{R}/sheet{k // 6}.jpg")
EOF
