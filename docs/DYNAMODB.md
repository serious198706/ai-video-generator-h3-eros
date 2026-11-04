# DynamoDB admission and task state

## Table to create

Use the existing AWS region **us-east-1**, one **DynamoDB Standard** table,
suggested name `h3-control`, **On-demand** billing:

| Setting | Value |
| --- | --- |
| Partition key | `pk`, String |
| Sort key | `sk`, String |
| TTL attribute | `expires_at` |
| Secondary indexes | None |
| Streams / global tables / DAX | Not required |

TTL is set only after a task finishes, default retention 30 days. Live tasks and
the capacity ledger never expire through TTL. AWS TTL deletion is asynchronous;
it is not used for lease expiry or capacity release. PITR is optional.

## Deployments

H3 checkout branch: `feat/dynamodb-admission`.
Existing API checkout (`../server`) branch: `feat/h3-dynamodb-admission`.
The deployed API is still `server/lambda_api`, at the existing URL. The H3 Lambda
sample is not the deployment target. API and worker must switch to DynamoDB
together, after the old queue has drained. There is no automatic import of old
S3 status JSON; pre-cutover tasks are not visible through the new status backend.
Old Wan workers must not consume the controlled H3 queue.

Configure **the same** values on the existing Lambda and each H3 worker:

```dotenv
H3_DYNAMODB_TABLE=h3-control
H3_CONTROL_NAMESPACE=h3
H3_CAPACITY_PROFILE=6000-s
H3_ADMISSION_LIMIT_SECONDS=780
H3_ADMISSION_MAX_TASKS=18
H3_WORKER_STALE_SECONDS=90
H3_TASK_LEASE_SECONDS=180
H3_TASK_RETENTION_SECONDS=2592000
```

Workers additionally require `H3_MODEL_LIFETIME=dual_gpu`, distinct
`H3_WORKER_ID` on every machine, and the existing SQS/S3 configuration.
`SQS_HEARTBEAT_SECONDS` must be less than half the task lease (default 60 vs 180).
Machines publish readiness every 20 seconds after model initialization, and expire
from available capacity after 90 seconds without a heartbeat. No configured
machine count is trusted: only ready, recent heartbeats count.

On the worker, existing `deploy.sh`, AWS setup and Supervisor start commands
continue to apply. Deployment checks the table schema before model download;
startup checks table read access before model loading. These checks do not prove
write permission: the IAM policy below is also required.

Build the **existing API** with `server/lambda_api/package.sh`. The script bundles
`minimax_h3/h3_control` and H3 request validation without model dependencies.
If repositories are not siblings, set `H3_SOURCE_DIR` to the H3 checkout. Deploy
the resulting ZIP to the existing Lambda and configure its environment. This
change does not create infrastructure, change live credentials, or deploy AWS
resources automatically.

When the table variable is empty, existing S3 status behavior is retained. Do not
change backends while tasks are outstanding: that would split status ownership.

## API contract

`POST /v1/generate` accepts 5/10/15 seconds, 720p/1080p, steps=4, audio=false in
controlled mode. Request validation runs before admission. Supply a stable,
unpredictable `Idempotency-Key` header (1..128 characters) for every business order.
The same key and request returns the same task, even when the pool is full.
Reusing a key for a different request returns 409. Keys share the control namespace;
the application backend should generate them, not expose cross-user predictable keys.
The key is retained as long as the task record remains in DynamoDB.

Successful admission returns 202 with the existing `id/task_id/status` fields,
plus `estimated_completion_seconds`, `dispatch_pending`, and `replayed`.
If SQS publication fails or times out, the accepted task remains durable and 202
is returned with `dispatch_pending=true`. Do **not** also submit that accepted task
to a2e; retry/query it using its existing identity.

When no machine is ready, unfinished tasks reach 18, or estimated completion
exceeds 780 seconds, the API returns 429 without creating a task or SQS message:

```json
{
  "detail": {
    "code": "LOCAL_CAPACITY_EXCEEDED",
    "estimated_completion_seconds": 821,
    "admission_limit_seconds": 780,
    "reason": "estimated_completion_exceeded",
    "fallback": "a2e"
  }
}
```

For `no_ready_workers`, the estimate is null. DynamoDB failures return 503, not a
misleading 429; a timeout may be an accepted write. Retry with the **same**
Idempotency-Key before deciding to send elsewhere. Without that header, automatic
retries cannot identify the original request.

`GET /v1/tasks/{id}` reads the authoritative DynamoDB state. Existing final
`video_url`, `result`, producer and compact timing fields remain. Full request
content is retained for recovery and identity checking, so API authentication
and authorization remain the responsibility of the existing gateway/backend.

## Capacity calculation

The pool uses measured homogeneous 6000-S (default) or Max-Q GPU1 generation
times. Per task it reserves GPU1 time plus 2 seconds for switching, with a final
10-second upload/control allowance and GPU0 super-resolution tail for 1080p.
GPU0 conditioning and super-resolution overlap with GPU1 generation.
Claimed jobs are attributed to their machine; waiting jobs are placed on the
least-loaded virtual machine. Fresh denoise progress reduces remaining work;
stale/offline claims are charged their full work again. 18 tasks is an additional
hard cap across queued and running work, not just visible SQS messages.

This predicts completion; it does **not guarantee a 15-minute SLA**. Standard SQS
does not guarantee FIFO ordering or balanced assignment. Hardware variation,
machine failure and AWS latency affect actual completion. Start with 780 seconds,
compare predicted vs actual times, and tune from production evidence. Increasing
the hard cap alone does not increase capacity. Do not mix profile speeds in one
namespace unless the estimator is extended for heterogeneous workers.

## Concurrency, recovery and artifact ownership

One bounded pool item contains worker heartbeats and unfinished reservations;
separate task items contain request/state, lease and SQS outbox payload. Versioned
DynamoDB transactions atomically reserve capacity and create a task, and atomically
persist terminal state and release capacity. Conditional retries handle competing
writers. Full task JSON is not copied into the pool ledger. Task records are
bounded before admission; this design targets a small GPU fleet, not an unbounded
high-throughput global scheduler.

Workers use a renewable lease with a unique attempt owner. A duplicate delivery
with an active lease is deferred, not deleted. Lease takeover fences old state
writes. S3 videos use immutable `<prefix>/<task_id>/<attempt_id>.mp4` object names;
only the current lease owner can publish the final key/URL. A crashed upload can
be recovered from its persisted key and fingerprint. Stale uploads can leave
unreferenced objects, so lifecycle cleanup needs a separate retention policy.

SQS visibility renewal, processing timeouts, callback retry rules and max-receive
handling remain. Capacity is released on terminal generation state; callback
completion does not unnecessarily occupy a GPU slot. An outstanding callback's
lease is retained until expiry so an immediate duplicate cannot race its delivery.

An unfinished job keeps its capacity reservation during an outage; it is not
silently dropped when a machine goes offline. SQS redelivery lets a healthy worker
take over after lease expiry. A manually deleted/DLQ task must be reconciled before
capacity is released. Never delete the pool ledger to clear a full queue.

## Outbox repair and diagnostics

Every live worker retries pending SQS publication every 20 seconds. For recovery
independent of GPU hosts, deploy the **same ZIP** as a small second Lambda with
handler `lambda_api.outbox.handler` and an EventBridge rate of one minute. Give it
the same table, namespace and SQS permissions. No GPU is needed. This is optional
when worker-based repair is sufficient, but recommended for production.

Commands below use already-exported environment variables and the service venv:

```bash
python -m h3_control.cli check
python -m h3_control.cli status
python -m h3_control.cli dispatch
```

`check` and `status` do not receive/delete SQS messages or write records. `dispatch`
explicitly republishes durable pending messages and marks them sent. Logs include
task/worker/run/attempt IDs but do not print prompts, input URLs or credentials.

## IAM additions

Add this statement to the existing Lambda execution role and GPU user/role,
replacing the table ARN with your actual table ARN. Existing SQS/S3 permissions
remain needed; ListBucket is not required by this change.

```json
{
  "Effect": "Allow",
  "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DescribeTable"],
  "Resource": "arn:aws:dynamodb:us-east-1:905641673118:table/h3-control"
}
```

DynamoDB authorizes transaction Put actions with `dynamodb:PutItem`; there is no
separate IAM `dynamodb:TransactWriteItems` action to grant. Workers now also require
`sqs:SendMessage` on the generation queue to repair the outbox. S3 permissions must
cover the existing prefix recursively because object names now include attempt
subdirectories. No AWS keys belong in this document or tracked configuration.

## Local simulation

Use the same code with real DynamoDB/SQS/S3 and no GPU through `sim.sh`. Set
`H3_CONTROL_NAMESPACE=h3-sim`, `H3_SIM_PROFILE=6000-s` and matching
`H3_CAPACITY_PROFILE=6000-s` in `.env.simulator`; use a separate simulation queue
and S3 prefix. Point a test API at the same simulation namespace. Submit through
the API: the old direct SQS simulator submit helper bypasses admission and is
intentionally rejected by a controlled worker. Three simulator processes need
distinct worker IDs, ports, work/log directories and service sockets.

## Cost correction

The earlier estimate of roughly $1 per 100k tasks assumed simple 1KB writes and
ten reads/writes per task. This implementation performs transactional writes of a
multi-KB ledger, full task records and regular lease heartbeats. Its actual cost
will be higher and depends on item size, traffic and polling. Use consumed capacity
and actual billing after the trial rather than treating the earlier estimate as
a deployment quote.

References: [DynamoDB transactions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html),
[TTL behavior](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html).
