#!/usr/bin/env python3
"""
load_to_dynamodb.py

Loads synthetic SAR JSON records (produced by generate_sar_data.py) into a
DynamoDB table for local/dev prototyping. Uses boto3's batch_writer for
efficient batched puts.

Usage:
    # against a real AWS account/region (dev/sandbox account only!)
    python3 load_to_dynamodb.py --table sar-prototype --file synthetic_sar_data.json

    # against a local DynamoDB (e.g. `docker run -p 8000:8000 amazon/dynamodb-local`)
    python3 load_to_dynamodb.py --table sar-prototype --file synthetic_sar_data.json \
        --endpoint-url http://localhost:8000

Table creation (if it doesn't already exist) mirrors the key design described
in dynamodb_schema.md:
    PK (partition key, S)   e.g. "SAR#<BSAID>"
    SK (sort key, S)        e.g. "METADATA"
    GSI1: GSI1PK / GSI1SK   -> query SARs by filing institution
    GSI2: GSI2PK / GSI2SK   -> query SARs by subject identifier
"""

import argparse
import json
import sys
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError


def to_dynamodb_json(obj):
    """DynamoDB's boto3 Table resource requires Decimal instead of float."""
    if isinstance(obj, list):
        return [to_dynamodb_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dynamodb_json(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return Decimal(str(obj))
    return obj


def ensure_table(ddb_client, table_name):
    try:
        ddb_client.describe_table(TableName=table_name)
        print(f"Table '{table_name}' already exists.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    print(f"Creating table '{table_name}'...")
    ddb_client.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
            {"AttributeName": "GSI2PK", "AttributeType": "S"},
            {"AttributeName": "GSI2SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1-InstitutionFilingDate",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            },
            {
                "IndexName": "GSI2-Subject",
                "KeySchema": [
                    {"AttributeName": "GSI2PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI2SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            },
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )
    waiter = ddb_client.get_waiter("table_exists")
    waiter.wait(TableName=table_name)
    print("Table created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="DynamoDB table name")
    parser.add_argument("--file", required=True, help="Path to synthetic_sar_data.json")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint-url", default=None, help="Use for DynamoDB Local")
    parser.add_argument("--create-table", action="store_true", help="Create table if missing")
    args = parser.parse_args()

    session = boto3.session.Session()
    client_kwargs = {"region_name": args.region}
    if args.endpoint_url:
        client_kwargs["endpoint_url"] = args.endpoint_url

    ddb_client = session.client("dynamodb", **client_kwargs)
    ddb_resource = session.resource("dynamodb", **client_kwargs)

    if args.create_table:
        ensure_table(ddb_client, args.table)

    table = ddb_resource.Table(args.table)

    with open(args.file) as f:
        records = json.load(f)

    with table.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as batch:
        for record in records:
            batch.put_item(Item=to_dynamodb_json(record))

    print(f"Loaded {len(records)} synthetic SAR records into '{args.table}'.")


if __name__ == "__main__":
    sys.exit(main())
