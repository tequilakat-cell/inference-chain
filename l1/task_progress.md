# Implementation Plan

## 1. SDK Improvements (client.py)
- [ ] Add NonceManager for concurrent transaction safety
- [ ] Add `register_miner()` method with RSA key support
- [ ] Add `register_model()` method
- [ ] Add `get_job_status()` / `get_job_output()` methods
- [ ] Add `get_miners_for_model()` method
- [ ] Add `token_stats()` endpoint accessor
- [ ] Add RSA key generation for requesters (encryption/decryption helpers)
- [ ] Add async-compatible inference method

## 2. API Server Improvements (server.py)
- [ ] Add nonce management per private key
- [ ] Add `GET /v1/jobs/{job_id}` endpoint
- [ ] Add `GET /v1/token-stats` endpoint
- [ ] Add `GET /v1/miners/{address}` endpoint
- [ ] Add `GET /v1/model/{model_id}` endpoint
- [ ] Add `POST /v1/miners/register` endpoint
- [ ] Add `POST /v1/models/register` endpoint
- [ ] Add `GET /v1/miners-for-model/{model_id}` endpoint
- [ ] Switch from blocking polling to async polling

## 3. Frontend Improvements (index.html)
- [ ] Add miner registration workflow
- [ ] Add submit output form for miners
- [ ] Add challenge result UI
- [ ] Fix job output display from event logs
- [ ] Add proper job detail view
- [ ] Improve overall UI/UX (better colors, animations, layout)
- [ ] Add real-time streaming-like job status updates
- [ ] Add responsive design improvements
- [ ] Fix the emission chart rendering
- [ ] Add toast notifications styling improvements
