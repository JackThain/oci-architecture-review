"""Quick check that your AWS credentials (outside this repo) work for Bedrock."""
import boto3

session = boto3.session.Session()  # reads ~/.aws/credentials and ~/.aws/config
region = session.region_name
print("Identity:", session.client("sts").get_caller_identity()["Arn"])
print("Region:  ", region or "NOT SET - export AWS_REGION or set it in ~/.aws/config")

if region:
    # Model access is granted per model, per region. A model missing from this
    # list is the usual cause of AccessDeniedException on the first call.
    models = session.client("bedrock").list_foundation_models()["modelSummaries"]
    print(f"Models visible in {region}: {len(models)}")
    for model in models[:10]:
        print("  ", model["modelId"])
