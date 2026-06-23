output "s3_bucket_name" {
  description = "S3 bucket for RAG data"
  value       = aws_s3_bucket.rag_data.bucket
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.rag_data.arn
}

output "athena_results_bucket" {
  description = "S3 bucket for Athena query results"
  value       = aws_s3_bucket.athena_results.bucket
}

output "knowledge_base_id" {
  description = "Bedrock Knowledge Base ID"
  value       = aws_bedrockagent_knowledge_base.rag.id
}

output "knowledge_base_data_source_id" {
  description = "Bedrock KB data source ID (for sync)"
  value       = aws_bedrockagent_data_source.s3_docs.data_source_id
}

output "opensearch_collection_endpoint" {
  description = "OpenSearch Serverless collection endpoint"
  value       = aws_opensearchserverless_collection.kb_vectors.collection_endpoint
}

output "glue_database_name" {
  description = "Glue catalog database name"
  value       = aws_glue_catalog_database.rag_db.name
}

output "athena_workgroup" {
  description = "Athena workgroup name"
  value       = aws_athena_workgroup.rag.name
}

output "pipeline_role_arn" {
  description = "IAM role ARN for the pipeline"
  value       = aws_iam_role.rag_pipeline.arn
}

output "developer_policy_arn" {
  description = "IAM policy ARN to attach to developer users"
  value       = aws_iam_policy.developer_rag_access.arn
}

output "bedrock_kb_role_arn" {
  description = "IAM role ARN used by the Bedrock Knowledge Base"
  value       = aws_iam_role.bedrock_kb.arn
}

# Helper commands
output "upload_docs_command" {
  description = "Upload source documents to S3"
  value       = "aws s3 sync ./your-documents/ s3://${aws_s3_bucket.rag_data.bucket}/source_documents/"
}

output "ingest_command" {
  description = "Run the Bedrock-powered ingestion with Parquet export"
  value       = "python -m document_ingestion.cli s3-ingest --s3-bucket ${aws_s3_bucket.rag_data.bucket} --workdir ./work --export-parquet"
}

output "kb_sync_command" {
  description = "Trigger KB sync after uploading new documents"
  value       = "python -m document_ingestion.cli kb-sync --data-source-id ${aws_bedrockagent_data_source.s3_docs.data_source_id} --kb-id ${aws_bedrockagent_knowledge_base.rag.id} --wait"
}

output "query_command" {
  description = "Query the RAG system (auto-routes KB vs Athena)"
  value       = "BEDROCK_KB_ID=${aws_bedrockagent_knowledge_base.rag.id} ATHENA_DATABASE=${aws_glue_catalog_database.rag_db.name} python -m document_ingestion.cli query -i"
}

output "env_vars" {
  description = "Environment variables to set for the pipeline"
  value       = <<-EOT
    export BEDROCK_KB_ID="${aws_bedrockagent_knowledge_base.rag.id}"
    export ATHENA_DATABASE="${aws_glue_catalog_database.rag_db.name}"
    export ATHENA_WORKGROUP="${aws_athena_workgroup.rag.name}"
    export S3_BUCKET="${aws_s3_bucket.rag_data.bucket}"
    export AWS_DEFAULT_REGION="${var.aws_region}"
  EOT
}
