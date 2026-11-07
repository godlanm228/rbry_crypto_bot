"""A passkey in software for the site's tests: what Face ID or Windows Hello would answer, with a real P-256 key."""

import hashlib
import json
import os

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from web.auth import b64url

RP, ORIGIN = "node.tail1234.ts.net", "https://node.tail1234.ts.net"


class SoftKey:
    """A passkey in software: what Face ID or Windows Hello would answer, signed with a real P-256 key."""

    def __init__(self, rp=RP, origin=ORIGIN):
        self.key, self.id, self.rp, self.origin, self.count = ec.generate_private_key(ec.SECP256R1()), os.urandom(16), rp, origin, 0

    def _client(self, kind, challenge):
        return json.dumps({"type": kind, "challenge": challenge, "origin": self.origin, "crossOrigin": False}).encode()

    def create(self, options_json):
        challenge = json.loads(options_json)["challenge"]
        pub = self.key.public_key().public_numbers()
        cose = cbor2.dumps({1: 2, 3: -7, -1: 1, -2: pub.x.to_bytes(32, "big"), -3: pub.y.to_bytes(32, "big")})
        auth_data = (hashlib.sha256(self.rp.encode()).digest() + bytes([0x45]) + (0).to_bytes(4, "big")
                     + bytes(16) + len(self.id).to_bytes(2, "big") + self.id + cose)
        return {"id": b64url(self.id), "rawId": b64url(self.id), "type": "public-key",
                "authenticatorAttachment": "platform", "clientExtensionResults": {},
                "response": {"clientDataJSON": b64url(self._client("webauthn.create", challenge)),
                             "attestationObject": b64url(cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})),
                             "transports": ["internal"]}}

    def get(self, options_json, uv=True):
        challenge = json.loads(options_json)["challenge"]
        self.count += 1
        auth_data = hashlib.sha256(self.rp.encode()).digest() + bytes([0x05 if uv else 0x01]) + self.count.to_bytes(4, "big")
        client = self._client("webauthn.get", challenge)
        sig = self.key.sign(auth_data + hashlib.sha256(client).digest(), ec.ECDSA(hashes.SHA256()))
        return {"id": b64url(self.id), "rawId": b64url(self.id), "type": "public-key",
                "authenticatorAttachment": "platform", "clientExtensionResults": {},
                "response": {"clientDataJSON": b64url(client), "authenticatorData": b64url(auth_data),
                             "signature": b64url(sig), "userHandle": b64url(b"rbry-owner")}}
