# Minimal footprint: one bucket, one role. Everything else in this project is
# either on the device or in a training job that exists for four minutes.
#
#   terraform init && terraform apply
#   terraform output -raw sagemaker_role_arn   -> paste into config/default.yaml

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "eu-west-2"
}

variable "bucket_name" {
  type        = string
  description = "must be globally unique -- no CHANGEME"
}

resource "aws_s3_bucket" "telemetry" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "telemetry" {
  bucket                  = aws_s3_bucket.telemetry.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "telemetry" {
  bucket = aws_s3_bucket.telemetry.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Vibration telemetry is worthless after a few months and storage costs are
# the thing that quietly ruins hobby AWS accounts.
resource "aws_s3_bucket_lifecycle_configuration" "telemetry" {
  bucket = aws_s3_bucket.telemetry.id
  rule {
    id     = "expire-raw"
    status = "Enabled"
    filter { prefix = "raw/" }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    expiration { days = 180 }
  }
}

resource "aws_iam_role" "sagemaker" {
  name = "${var.bucket_name}-sagemaker"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Scoped to this bucket only. The AWS-managed SageMakerFullAccess policy grants
# s3:* on every bucket in the account, which is not a trade I'm making for a
# project that reads one prefix.
resource "aws_iam_role_policy" "sagemaker_s3" {
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.telemetry.arn, "${aws_s3_bucket.telemetry.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.telemetry.arn}/models/*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup", "logs:DescribeLogStreams"]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

output "bucket" { value = aws_s3_bucket.telemetry.id }
output "sagemaker_role_arn" { value = aws_iam_role.sagemaker.arn }
