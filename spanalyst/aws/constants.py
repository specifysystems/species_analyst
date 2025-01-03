"""Constants for Specify Network AWS Resources."""

# .............................................................................
# AWS constants
# .............................................................................
# TODO: Move base project-specific vars to a configuration file
PROJECT = "specnet"
REGION = "us-east-1"
AWS_ACCOUNT = "321942852011"

TASK_ROLE_NAME = f"{PROJECT}_task_role"
WORKFLOW_ROLE_NAME = f"{PROJECT}_workflow_role"
WORKFLOW_ROLE_ARN = f"arn:aws:iam::{PROJECT}:role/service-role/{WORKFLOW_ROLE_NAME}"

# AWS-defined
AWS_METADATA_URL = "http://169.254.169.254/latest/"
PARQUET_EXT = ".parquet"

GBIF_BUCKET = f"gbif-open-data-{REGION}"
GBIF_ARN = f"arn:aws:s3:::{GBIF_BUCKET}"
GBIF_ODR_FNAME = f"occurrence{PARQUET_EXT}"

# Instance types: https://aws.amazon.com/ec2/spot/pricing/
EC2_INSTANCE_TYPE = "t4g.micro"
EC2_SPOT_TEMPLATE = f"{PROJECT}_spot_task_template"

S3_BUCKET = f"{PROJECT}-{AWS_ACCOUNT}-{REGION}"
S3_IN_DIR = "input"
S3_OUT_DIR = "output"
S3_LOG_DIR = "log"
S3_SUMMARY_DIR = "summary"
S3_RS_TABLE_SUFFIX = f"_000{PARQUET_EXT}"
