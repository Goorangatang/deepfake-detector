---
name: ML model server deployment target
description: Why a TensorFlow/Keras (or similar heavy-model) server must deploy as Reserved VM, not Autoscale, on Replit.
---

# Deploy heavy ML model servers as Reserved VM, not Autoscale

A Python server that loads a large ML framework (TensorFlow/Keras/PyTorch) and/or
downloads model weights at runtime must use the **Reserved VM** deployment target,
not Autoscale.

**Why:**
- Autoscale promotes a build only after a startup HTTP probe returns 200 within a
  deadline. A single worker gets blocked while `import tensorflow` initializes
  in-process (C-extension init holds the GIL) and while the model downloads from a
  hub (e.g. Hugging Face) on first run — so the probe gets 500/timeout and the
  promote step fails even though the build succeeded.
- Autoscale scales to zero: between requests the instance is frozen/killed, which
  kills any background model-loading thread, and the ephemeral filesystem means the
  model re-downloads on every cold start. An always-loaded inference server is
  fundamentally a long-running process → Reserved VM.

**How to apply:**
- The deployment type is set by the user in the Publishing/Deployments pane
  (Advanced). It CANNOT be changed in code/artifact.toml — instruct the user to
  switch Autoscale → Reserved VM, then re-publish.
- Symptom signature: build logs end at "Creating Autoscale service"; runtime logs
  show repeated `healthcheck / returned status 500` right after gunicorn binds.
- The production run command binding `0.0.0.0:$PORT` is already correct; no code
  change is needed — only the deployment target.
