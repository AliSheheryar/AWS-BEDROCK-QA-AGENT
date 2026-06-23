variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "doc-rag"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for RAG documents and pipeline output"
  type        = string
}

variable "glue_database_name" {
  description = "Glue Data Catalog database name for Athena"
  type        = string
  default     = "doc_rag_db"
}

variable "athena_workgroup_name" {
  description = "Athena workgroup name"
  type        = string
  default     = "doc-rag-workgroup"
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model ID for generation"
  type        = string
  default     = "anthropic.claude-3-5-sonnet-20241022-v2:0"
}
