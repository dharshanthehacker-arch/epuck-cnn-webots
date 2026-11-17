#!/bin/zsh
# Live view of what Claude is doing in the background for this project.
P="${0:A:h:h}"
while true; do
  clear
  print -P "%B=== Webots CNN project - live status ($(date +%H:%M:%S)) ===%b\n"
  print -P "%B[1] CNN training (Jupyter kernel)%b"
  ps -Ao pid,pcpu,etime,command | grep "[i]pykernel_launcher" | awk '{printf "    running: CPU %s%%  elapsed %s\n", $2, $3}' || true
  pgrep -f ipykernel_launcher >/dev/null || echo "    idle"
  print -P "\n%B[2] Notebook results%b"
  ls -1 "$P/notebook/results" 2>/dev/null | sed 's/^/    /'
  print -P "\n%B[3] Trained models%b"
  ls -lh "$P/notebook/models" 2>/dev/null | awk 'NR>1{print "    "$NF"  "$5}'
  print -P "\n%B[4] Figures for the slides%b"
  ls -1 "$P/notebook/figures" 2>/dev/null | sed 's/^/    /'
  print -P "\n%B[5] Webots captures%b"
  for d in tour scene demo_frames; do
    n=$(ls "$P/captures/$d" 2>/dev/null | wc -l | tr -d ' ')
    echo "    captures/$d: $n files"
  done
  pgrep -f "Webots-R2025a" >/dev/null && echo "    Webots: RUNNING" || echo "    Webots: not running"
  print -P "\n%B[6] Git history%b"
  git -C "$P" log --oneline -8 2>/dev/null | sed 's/^/    /'
  print -P "\n%B[7] Presentation%b"
  ls -lh "$P/presentation"/*.pptx 2>/dev/null | awk '{print "    "$NF"  "$5}' || echo "    not built yet"
  sleep 5
done
