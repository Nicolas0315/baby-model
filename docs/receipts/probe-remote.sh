set -u
date -u +"probe_at=%FT%TZ"
PID=$(pgrep -f "bin/python -m baby_model.minigrid_torch_sweep" | head -1)
if [ -z "$PID" ]; then echo "STATE=no_process"; else
  echo "STATE=running PID=$PID"
  ps -o etime=,rss=,args= -p "$PID" | cut -c1-150
  awk '{print "cpu_ticks_utime="$14" stime="$15}' /proc/$PID/stat
fi
if [ -f "$HOME/v254.log" ]; then stat -c "log_bytes=%s log_mtime=%y" "$HOME/v254.log"; cat "$HOME/v254.log"; else echo "STATE=no_log"; fi
ls -d "$HOME/work/baby-model-cuda-56f0c3d/.tmp/v254-gotolocal" 2>/dev/null && echo "artifact_dir=present" || echo "artifact_dir=absent (written only at sweep end)"
