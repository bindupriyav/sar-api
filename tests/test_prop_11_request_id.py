# Feature: sar-classification-agent, Property 11: when an X-Request-Id header is present, result.requestId equals the supplied value
"""Property 11 for the SAR Classification Agent ``POST /classify`` endpoint.


Property statement: when an ``X-Request-Id`` header is
present on a well-formed ``POST /classify`` request, the returned
``Classification_Result`` echoes that exact value in its ``requestId`` field for
request correlation. When the header is absent, ``requestId`` is null.

Test approach
-------------
The endpoint is driven end-to-end through :class:`fastapi.testclient.TestClient`.
Importing ``app.main`` constructs a boto3 DynamoDB resource lazily (no network
call at import time), so ``from app.main import app`` works fully offline, and the
transaction classification path never touches DynamoDB.

To stay on the transaction path (and avoid the BSAID 501 placeholder) every
request carries a valid ``transactions`` payload only — two sub-threshold
``CASH_DEPOSIT`` amounts one day apart — and no ``bsaId``. Hypothesis generates
arbitrary header-safe, non-empty ``X-Request-Id`` values and asserts the response
is 200 and ``requestId`` equals the supplied value. An example-based test covers
the header-absent case (``requestId`` is null).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# A valid two-transaction payload that classifies via the transaction path
# without ever hitting DynamoDB (no bsaId). Both amounts are sub-threshold cash
# deposits one day apart.
_VALID_PAYLOAD = {
    "transactions": [
        {"amount": "9999.99", "date": "2024-01-01", "transactionType": "CASH_DEPOSIT"},
        {"amount": "9999.99", "date": "2024-01-02", "transactionType": "CASH_DEPOSIT"},
    ]
}


# Header-safe X-Request-Id generator: non-empty, printable ASCII with no control
# characters (HTTP header values must be a single line of visible ASCII, so we
# exclude CR/LF and other control chars and keep the byte range 0x21..0x7E).
_REQUEST_IDS = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=128,
)


@settings(max_examples=100, deadline=None)
@given(request_id=_REQUEST_IDS)
def test_request_id_echoed_when_header_present(request_id: str):
    response = client.post(
        "/classify",
        json=_VALID_PAYLOAD,
        headers={"X-Request-Id": request_id},
    )

    assert response.status_code == 200
    body = response.json()
    # Req 1.4: the supplied X-Request-Id is echoed verbatim into requestId.
    assert body["requestId"] == request_id


def test_request_id_null_when_header_absent():
    response = client.post("/classify", json=_VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    # No X-Request-Id header supplied -> requestId is null (Req 1.4).
    assert body["requestId"] is None
