import base64
from app import PROFILE_ID, app, canonical_json_bytes, compute_proposal_digest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

client = TestClient(app)


def b64url_encode(b: bytes) -> str:
  return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def run_tests():
  priv = Ed25519PrivateKey.generate()
  pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
  jwk = {"kty": "OKP", "crv": "Ed25519", "x": b64url_encode(pub_bytes)}

  propose_payload = {
      "profile": PROFILE_ID,
      "operation": "propose",
      "evaluationId": "eval_test_001",
      "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": jwk},
      "corpus": {
          "coreId": "c1",
          "auditId": "a1",
          "stableCount": 1,
          "freshCount": 0,
      },
      "allowedActions": [
          "create_draft",
          "update_internal_record",
          "send_approved_notice",
          "request_confirmation",
          "quarantine_item",
          "no_action",
      ],
      "dossiers": [{
          "dossierId": "dos_1",
          "partition": "stable_core",
          "receivedAt": "2026-10-31T00:00:00Z",
          "mailbox": "support",
          "objective": "Process incoming mail",
          "sources": [{
              "sourceId": "src_1",
              "kind": "email",
              "provenance": "internal",
              "title": "FYI Memo",
              "lines": [
                  {
                      "lineId": "L1",
                      "text": "Informational update for REF-9981.",
                  }
              ],
          }],
      }],
  }

  # 1. Initial Propose
  r1 = client.post("/mailroom", json=propose_payload)
  assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
  data1 = r1.json()
  assert data1["status"] == "awaiting_receipts"

  # 2. Exact Propose Replay -> 200 byte-identical
  r2 = client.post("/mailroom", json=propose_payload)
  assert r2.status_code == 200 and r2.content == r1.content

  # 3. Changed content with same evaluationId -> 409 Conflict
  mutated = dict(propose_payload)
  mutated["corpus"] = {
      "coreId": "c2",
      "auditId": "a1",
      "stableCount": 1,
      "freshCount": 0,
  }
  r3 = client.post("/mailroom", json=mutated)
  assert r3.status_code == 409, f"Expected 409, got {r3.status_code}"

  # 4. Build Valid Signed Receipt & Test Tampered Signature
  prop = data1["proposals"][0]
  p_digest = compute_proposal_digest(prop)
  inner_receipt = {
      "dossierId": prop["dossierId"],
      "callId": prop["callId"],
      "action": prop["action"],
      "accepted": True,
      "proposalDigest": p_digest,
      "receiptId": "rcpt_nonce_001",
  }
  sign_payload = {
      "profile": PROFILE_ID,
      "evaluationId": "eval_test_001",
      "inputDigest": data1["inputDigest"],
      "receipt": inner_receipt,
  }
  sig = base64.b64encode(priv.sign(canonical_json_bytes(sign_payload))).decode(
      "ascii"
  )

  # Tampered receipt test -> 400
  bad_commit = {
      "profile": PROFILE_ID,
      "operation": "commit",
      "evaluationId": "eval_test_001",
      "inputDigest": data1["inputDigest"],
      "receipts": [{**inner_receipt, "accepted": False, "receiptSignature": sig}],
  }
  r_bad = client.post("/mailroom", json=bad_commit)
  assert r_bad.status_code == 400, f"Expected 400 on tampered receipt, got {r_bad.status_code}"

  # 5. Valid Commit -> 200 executed
  good_commit = {
      "profile": PROFILE_ID,
      "operation": "commit",
      "evaluationId": "eval_test_001",
      "inputDigest": data1["inputDigest"],
      "receipts": [{**inner_receipt, "receiptSignature": sig}],
  }
  r_good = client.post("/mailroom", json=good_commit)
  assert r_good.status_code == 200
  assert r_good.json()["outcomes"][0]["status"] == "executed"

  # 6. Exact Commit Replay -> 200 byte-identical
  r_replay = client.post("/mailroom", json=good_commit)
  assert r_replay.status_code == 200 and r_replay.content == r_good.content
  print("All 6 local protocol & cryptographic checks passed!")


if __name__ == "__main__":
  run_tests()