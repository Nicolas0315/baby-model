# v2.54 execution receipt (provenance partially reconstructed after the fact)

- tarball: bm-v254.tar.gz
- tarball_sha256: 0391ae5bdccbcaa2623f41311fb5c634a7a3215dc081b0902b555418d9914616
- source commit that produced the tarball: dcae02768... (full below)
- full_sha: dcae02768fcc4cc789ddda29f2bce7dd06fa5530
- BABY_MODEL_SOURCE_COMMIT recorded in the artifacts: 'v254' (a LABEL, not a SHA -- this is the defect the audit flagged)
- remote start receipt: 2026-08-23T16:23:38Z (from $HOME/v254.log)
- remote liveness probes: PID 199783, cpu_utime 136168 -> 142294 over 61s, rss 834MB
- remote end: log_mtime 2026-08-24 08:08:08 +0900, artifact_dir present

Limitation: the SHA linkage above is reconstructed locally after the run, not
recorded by the run itself. Future runs must write the full SHA and the tarball
SHA-256 into the artifact at execution time.
# v2.54 remote liveness receipt
captured_locally_at=2026-08-24T00:26:52Z
captured_by=CA-20032518
--- probe 1 ---
probe_at=2026-08-24T00:26:53Z
STATE=no_process
log_bytes=223 log_mtime=2026-08-24 08:08:08.330461742 +0900
start 2026-08-23T16:23:38Z
episode_rows=142384
minigrid_torch_sweep_dir=.tmp/v254-gotolocal/20260823T230807Z
winner_by_mean_success_last_window=ZK_torch_gotoobj_curriculum_no_repr_delay_long
exit=0
end 2026-08-23T23:08:08Z
/home/ogosh/work/baby-model-cuda-56f0c3d/.tmp/v254-gotolocal
artifact_dir=present
--- sleeping 60s ---
--- probe 2 ---
probe_at=2026-08-24T00:27:54Z
STATE=no_process
log_bytes=223 log_mtime=2026-08-24 08:08:08.330461742 +0900
start 2026-08-23T16:23:38Z
episode_rows=142384
minigrid_torch_sweep_dir=.tmp/v254-gotolocal/20260823T230807Z
winner_by_mean_success_last_window=ZK_torch_gotoobj_curriculum_no_repr_delay_long
exit=0
end 2026-08-23T23:08:08Z
/home/ogosh/work/baby-model-cuda-56f0c3d/.tmp/v254-gotolocal
artifact_dir=present
