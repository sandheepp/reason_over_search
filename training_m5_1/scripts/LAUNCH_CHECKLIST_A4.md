# M5.1-prod-a4 launch checklist

**Do not start training until every box below is checked.** This list was authored 2026-05-15 in direct response to the M5.1-prod-a3 incident (Spheron spot preempted at step 56; uploader silently dropped 38 rollouts and the step_50 checkpoint).

## Pre-launch (run on the new instance, before any training)

- [ ] HF_TOKEN set in env (`echo ${HF_TOKEN:0:8}...`)
- [ ] HF_REPO_ID set to the **new** repo name (e.g. `pantomiman/qwen3.5-0.8b-grpo-musique-b200-a4-seed42`)
- [ ] HF_BACKUP_REPO_ID set to a second namespace (e.g. `pantomiman/qwen3.5-0.8b-grpo-musique-b200-a4-backup`)
- [ ] BOTH repos created on HF (visit https://huggingface.co/new repo, set private, create)
- [ ] `pip install huggingface_hub hf_transfer` in miniforge3 base (`/opt/miniforge3/bin/pip install ...`)
- [ ] Run preflight test: `python3 upload_a4_to_hf.py --skip-preflight=false &` and wait 30s; check `uploader.log` has `preflight: ✓ canary uploaded` for BOTH repos
- [ ] `.uploader_canary.txt` exists on BOTH HF repos (visit them in browser)

## External monitor (run on YOUR local machine, before launching training)

- [ ] Run `python3 external_monitor.py <repo> --wandb-run <id>` once — should report HEALTHY for the preflight canary
- [ ] Set up a cron / launchd job that re-runs the monitor every 5 min and notifies you on STALE
  - macOS launchd example: `~/Library/LaunchAgents/com.somedude.uploadmon.plist` calling the script every 300s
  - Slack/email webhook on non-zero exit
- [ ] Verify the cron fires once (touch the log file's mtime forward; should see a new line)

## During-run sanity (every cadence checkpoint)

- [ ] Before writing the cadence doc, grep uploader.log: `tail -50 /workspace/uploader.log | grep heartbeat` — should show at least 5 heartbeats in the last 10 min
- [ ] Verify HF rollout step count matches training step count ±1: external_monitor output should show `lag <= 2`
- [ ] If lag > 5, STOP cadence doc work and diagnose uploader first
- [ ] First ckpt (step_50): verify it appears in HF within 5 min of NeMo-RL renaming tmp_step_50 → step_50; if not in 10 min, hard stop

## Hard stop conditions (kill training and investigate)

- External monitor reports STALE for 2+ consecutive polls (10+ min)
- HF rollout lag > 10 steps
- uploader.log has no heartbeats in 5 min
- Step ckpt (step_50, step_100, …) doesn't appear on HF within 10 min of save
- Either repo returns 403/404 (auth or repo issue)

## What "good" looks like in uploader.log

```
[2026-05-15T08:00:00Z] [INFO] uploader_a4 starting; primary=pantomiman/...-a4-seed42 backup=pantomiman/...-a4-backup
[2026-05-15T08:00:02Z] [INFO] preflight: testing upload to pantomiman/...-a4-seed42
[2026-05-15T08:00:05Z] [INFO] preflight: ✓ canary uploaded
[2026-05-15T08:00:08Z] [INFO] preflight: ✓ canary uploaded
[2026-05-15T08:00:08Z] [INFO] uploading to 2 repos
[2026-05-15T08:01:08Z] [INFO] [heartbeat] cycle=1 uploaded_total=0 new_this_cycle=0 queue_pending=0 errors=0 elapsed=0.3s
[2026-05-15T08:17:42Z] [INFO]   → uploading folder step_50 from /workspace/results/.../step_50
[2026-05-15T08:17:52Z] [INFO]   ✓ step_50 (folder)
[2026-05-15T08:17:53Z] [INFO]   ✓ step_50 (folder)         ← second is to backup repo
[2026-05-15T08:18:08Z] [INFO]   ✓ exp_013/train_data_step50.jsonl
[2026-05-15T08:18:09Z] [INFO]   ✓ exp_013/train_data_step50.jsonl
[2026-05-15T08:18:08Z] [INFO] [heartbeat] cycle=18 uploaded_total=4 new_this_cycle=4 queue_pending=0 errors=0 elapsed=15.7s
```

## What "broken" looks like

- Heartbeats stop appearing for >2 min
- `[STALE]` lines start appearing
- `errors > 0` in heartbeats for 3+ consecutive cycles
- queue_pending grows without bound

## Recovery if uploader breaks mid-run

1. `tail /workspace/uploader.log` — what's the last successful line?
2. `ps -ef | grep upload_a4` — is the process alive?
3. `cat /workspace/.uploaded_artifacts.json` — what does state think is uploaded?
4. **Run the external monitor** — what does HF actually have?
5. If process dead: restart it manually with same env vars; on launch it scans state and resumes
6. If process alive but stuck: send SIGTERM and let it shutdown gracefully, then restart
7. If both broken: continue training (it doesn't depend on the uploader) but ALL artifacts are now at-risk; copy step_N/ from disk to local laptop via scp every 50 steps as a manual backup

## Notes for future-me

- The bash uploader (`upload_a3_to_hf.sh`) is kept in the repo as a record of the bug, but DO NOT use it
- The python uploader (`upload_a4_to_hf.py`) is what runs on prod
- New runs go in `a4`, `a5`, … namespaces; never reuse `a3` because the repo has the broken artifacts
- Cron the external monitor from a machine that is NOT the training instance — that's the whole point
