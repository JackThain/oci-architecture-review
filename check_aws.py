"""Quick check that your AWS credentials (outside this repo) work for Bedrock.

Prints the exact IDs you can paste into AWS_MODEL_ID in .env.
"""
import boto3

session = boto3.session.Session()  # reads ~/.aws/credentials and ~/.aws/config
region = session.region_name
print("Identity:", session.client("sts").get_caller_identity()["Arn"])
print("Region:  ", region or "NOT SET - set AWS_REGION or run: aws configure")

if not region:
    raise SystemExit("No region configured, so there is nothing to list.")

bedrock = session.client("bedrock")

# Model access is granted per model, per region. A model you have not enabled
# in the Bedrock console will not appear here, which is the usual cause of
# AccessDeniedException on the first call.
models = bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
print(f"\nText models visible in {region}: {len(models)}")

on_demand, profile_only = [], []
for model in models:
    types = model.get("inferenceTypesSupported", [])
    if "ON_DEMAND" in types:
        on_demand.append(model["modelId"])
    elif "INFERENCE_PROFILE" in types:
        profile_only.append(model["modelId"])

print(f"\n  Callable directly as AWS_MODEL_ID ({len(on_demand)}):")
for model_id in on_demand:
    print("    ", model_id)

if profile_only:
    print(f"\n  Require an inference profile ID instead ({len(profile_only)}):")
    for model_id in profile_only:
        print("    ", model_id)

# Cross-region inference profiles carry a us. / eu. / apac. prefix. For any
# model listed above as profile-only, the profile ID is what Converse expects.
profiles = bedrock.list_inference_profiles().get("inferenceProfileSummaries", [])
active = [p for p in profiles if p.get("status") == "ACTIVE"]
print(f"\n  Inference profiles ({len(active)} active) - use these IDs verbatim:")
for profile in active:
    print(f"     {profile['inferenceProfileId']}   ({profile['type']})")

if not on_demand and not active:
    print("\nNothing is callable yet. Open the Bedrock console in this region,")
    print("go to Model access, and enable the models you want.")
