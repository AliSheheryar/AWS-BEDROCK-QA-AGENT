terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ====================================================================
# S3 Buckets
# ====================================================================

resource "aws_s3_bucket" "rag_data" {
  bucket = var.s3_bucket_name
}

resource "aws_s3_bucket_versioning" "rag_data" {
  bucket = aws_s3_bucket.rag_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "rag_data" {
  bucket = aws_s3_bucket.rag_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "rag_data" {
  bucket                  = aws_s3_bucket.rag_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "rag_data" {
  bucket = aws_s3_bucket.rag_data.id

  rule {
    id     = "archive-old-timelines"
    status = "Enabled"
    filter {
      prefix = "timelines/"
    }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_object" "source_docs_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "source_documents/"
  content = ""
}

resource "aws_s3_object" "raw_md_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "raw_markdown/"
  content = ""
}

resource "aws_s3_object" "clean_md_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "clean_markdown/"
  content = ""
}

resource "aws_s3_object" "database_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "database/"
  content = ""
}

resource "aws_s3_object" "timelines_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "timelines/"
  content = ""
}

resource "aws_s3_object" "athena_data_prefix" {
  bucket  = aws_s3_bucket.rag_data.id
  key     = "athena_data/"
  content = ""
}

# Athena query results bucket
resource "aws_s3_bucket" "athena_results" {
  bucket = "${var.s3_bucket_name}-athena-results"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "athena_results" {
  bucket                  = aws_s3_bucket.athena_results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  rule {
    id     = "expire-query-results"
    status = "Enabled"
    filter {
      prefix = ""
    }
    expiration {
      days = 7
    }
  }
}

# ====================================================================
# Glue Data Catalog (for Athena)
# ====================================================================

resource "aws_glue_catalog_database" "rag_db" {
  name = var.glue_database_name
}

resource "aws_glue_catalog_table" "events" {
  name          = "events"
  database_name = aws_glue_catalog_database.rag_db.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "parquet"
    EXTERNAL         = "TRUE"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.rag_data.bucket}/athena_data/events/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
      parameters = {
        "serialization.format" = "1"
      }
    }

    columns {
      name = "event_id"
      type = "int"
    }
    columns {
      name = "date"
      type = "string"
    }
    columns {
      name = "action"
      type = "string"
    }
    columns {
      name = "party"
      type = "string"
    }
    columns {
      name = "court"
      type = "string"
    }
    columns {
      name = "source_documents"
      type = "string"
    }
    columns {
      name = "excerpt"
      type = "string"
    }
    columns {
      name = "other_actors"
      type = "string"
    }
    columns {
      name = "role_gate"
      type = "string"
    }
    columns {
      name = "canonical_actor_id"
      type = "string"
    }
    columns {
      name = "confidence"
      type = "string"
    }
    columns {
      name = "forum_id"
      type = "string"
    }
    columns {
      name = "forum_role"
      type = "string"
    }
    columns {
      name = "raw_action"
      type = "string"
    }
    columns {
      name = "grounding_excerpt"
      type = "string"
    }
  }
}

resource "aws_glue_catalog_table" "case_actors" {
  name          = "case_actors"
  database_name = aws_glue_catalog_database.rag_db.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "parquet"
    EXTERNAL         = "TRUE"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.rag_data.bucket}/athena_data/case_actors/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
      parameters = {
        "serialization.format" = "1"
      }
    }

    columns {
      name = "name"
      type = "string"
    }
    columns {
      name = "role"
      type = "string"
    }
    columns {
      name = "mentions"
      type = "int"
    }
    columns {
      name = "canonical_id"
      type = "string"
    }
  }
}

# ====================================================================
# Athena Workgroup
# ====================================================================

resource "aws_athena_workgroup" "rag" {
  name = var.athena_workgroup_name

  configuration {
    enforce_workgroup_configuration = true
    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results.bucket}/results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
  }
}

# ====================================================================
# OpenSearch Serverless (vector store for Bedrock Knowledge Base)
# ====================================================================

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.project_name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource      = ["collection/${var.project_name}-kb-vectors"]
      }
    ]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.project_name}-net"
  type = "network"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "collection"
        Resource      = ["collection/${var.project_name}-kb-vectors"]
      },
      {
        ResourceType = "dashboard"
        Resource      = ["collection/${var.project_name}-kb-vectors"]
      }
    ]
    AllowFromPublic = true
  }])
}

resource "aws_opensearchserverless_collection" "kb_vectors" {
  name = "${var.project_name}-kb-vectors"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

resource "aws_opensearchserverless_access_policy" "kb_access" {
  name = "${var.project_name}-kb-access"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "collection"
        Resource      = ["collection/${var.project_name}-kb-vectors"]
        Permission = [
          "aoss:CreateCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems"
        ]
      },
      {
        ResourceType = "index"
        Resource      = ["index/${var.project_name}-kb-vectors/*"]
        Permission = [
          "aoss:CreateIndex",
          "aoss:UpdateIndex",
          "aoss:DescribeIndex",
          "aoss:ReadDocument",
          "aoss:WriteDocument"
        ]
      }
    ]
    Principal = [
      aws_iam_role.bedrock_kb.arn,
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
    ]
  }])
}

# ====================================================================
# Bedrock Knowledge Base
# ====================================================================

resource "aws_iam_role" "bedrock_kb" {
  name = "${var.project_name}-bedrock-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_s3" {
  name = "s3-access"
  role = aws_iam_role.bedrock_kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.rag_data.arn,
          "${aws_s3_bucket.rag_data.arn}/clean_markdown/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_model" {
  name = "bedrock-model"
  role = aws_iam_role.bedrock_kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "bedrock:InvokeModel"
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_aoss" {
  name = "opensearch-access"
  role = aws_iam_role.bedrock_kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "aoss:APIAccessAll"
        Resource = aws_opensearchserverless_collection.kb_vectors.arn
      }
    ]
  })
}

resource "aws_bedrockagent_knowledge_base" "rag" {
  name     = "${var.project_name}-knowledge-base"
  role_arn = aws_iam_role.bedrock_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.kb_vectors.arn
      vector_index_name = "bedrock-knowledge-base-default-index"
      field_mapping {
        vector_field   = "bedrock-knowledge-base-default-vector"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        metadata_field = "AMAZON_BEDROCK_METADATA"
      }
    }
  }

  depends_on = [
    aws_opensearchserverless_access_policy.kb_access,
    aws_iam_role_policy.bedrock_kb_aoss,
  ]
}

resource "aws_bedrockagent_data_source" "s3_docs" {
  name                 = "clean-markdown-docs"
  knowledge_base_id    = aws_bedrockagent_knowledge_base.rag.id
  data_deletion_policy = "RETAIN"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn              = aws_s3_bucket.rag_data.arn
      inclusion_prefixes      = ["clean_markdown/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = 512
        overlap_percentage = 20
      }
    }
  }
}

# ====================================================================
# IAM: Pipeline Role (for running ingestion + queries)
# ====================================================================

resource "aws_iam_role" "rag_pipeline" {
  name = "${var.project_name}-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "rag_s3_access" {
  name = "s3-access"
  role = aws_iam_role.rag_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.rag_data.arn,
          "${aws_s3_bucket.rag_data.arn}/*",
          aws_s3_bucket.athena_results.arn,
          "${aws_s3_bucket.athena_results.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "rag_bedrock_access" {
  name = "bedrock-access"
  role = aws_iam_role.rag_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockInvoke"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-5-sonnet-*",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-sonnet-*",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-haiku-*",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
        ]
      },
      {
        Sid    = "BedrockKB"
        Effect = "Allow"
        Action = [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate"
        ]
        Resource = aws_bedrockagent_knowledge_base.rag.arn
      },
      {
        Sid    = "BedrockKBSync"
        Effect = "Allow"
        Action = [
          "bedrock:StartIngestionJob",
          "bedrock:GetIngestionJob",
          "bedrock:ListIngestionJobs"
        ]
        Resource = aws_bedrockagent_knowledge_base.rag.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "rag_athena_access" {
  name = "athena-access"
  role = aws_iam_role.rag_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AthenaQuery"
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:StopQueryExecution"
        ]
        Resource = aws_athena_workgroup.rag.arn
      },
      {
        Sid    = "GlueCatalog"
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetPartition",
          "glue:GetPartitions"
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:database/${var.glue_database_name}",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.glue_database_name}/*"
        ]
      }
    ]
  })
}

# ====================================================================
# IAM: Developer access policy (attach to IAM user for local dev)
# ====================================================================

resource "aws_iam_policy" "developer_rag_access" {
  name        = "${var.project_name}-developer-access"
  description = "Policy for developers running the RAG pipeline locally"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.rag_data.arn,
          "${aws_s3_bucket.rag_data.arn}/*",
          aws_s3_bucket.athena_results.arn,
          "${aws_s3_bucket.athena_results.arn}/*"
        ]
      },
      {
        Sid    = "BedrockInvoke"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.*"
      },
      {
        Sid    = "BedrockKB"
        Effect = "Allow"
        Action = [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate",
          "bedrock:StartIngestionJob",
          "bedrock:GetIngestionJob",
          "bedrock:ListIngestionJobs"
        ]
        Resource = aws_bedrockagent_knowledge_base.rag.arn
      },
      {
        Sid    = "Athena"
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:StopQueryExecution"
        ]
        Resource = aws_athena_workgroup.rag.arn
      },
      {
        Sid    = "GlueCatalog"
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetPartition",
          "glue:GetPartitions"
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:database/${var.glue_database_name}",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.glue_database_name}/*"
        ]
      }
    ]
  })
}
