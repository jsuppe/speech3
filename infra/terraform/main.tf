# SpeakFit Cloud Infrastructure
# Provider: Google Cloud Platform

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  # Remote state (uncomment for production)
  # backend "gcs" {
  #   bucket = "speakfit-terraform-state"
  #   prefix = "terraform/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "domain" {
  description = "Domain name for the API"
  type        = string
  default     = "api.speakfit.app"
}

# Scaling configuration
variable "api_min_instances" {
  default = 1
}

variable "api_max_instances" {
  default = 5
}

variable "gpu_worker_count" {
  description = "Number of GPU workers"
  default     = 1
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

resource "google_compute_network" "vpc" {
  name                    = "speakfit-${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "speakfit-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  private_ip_google_access = true
}

# Cloud NAT for outbound internet from private instances
resource "google_compute_router" "router" {
  name    = "speakfit-${var.environment}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name   = "speakfit-${var.environment}-nat"
  router = google_compute_router.router.name
  region = var.region
  
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Firewall rules
resource "google_compute_firewall" "allow_http" {
  name    = "speakfit-${var.environment}-allow-http"
  network = google_compute_network.vpc.name
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["api-server"]
}

resource "google_compute_firewall" "allow_internal" {
  name    = "speakfit-${var.environment}-allow-internal"
  network = google_compute_network.vpc.name
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  source_ranges = ["10.0.0.0/24"]
}

# -----------------------------------------------------------------------------
# Cloud SQL (PostgreSQL)
# -----------------------------------------------------------------------------

resource "google_sql_database_instance" "postgres" {
  name             = "speakfit-${var.environment}-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier = var.environment == "prod" ? "db-custom-4-16384" : "db-custom-2-8192"
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }
    
    backup_configuration {
      enabled            = true
      start_time         = "03:00"
      binary_log_enabled = false
      
      backup_retention_settings {
        retained_backups = 7
      }
    }
    
    insights_config {
      query_insights_enabled = true
    }
  }
  
  deletion_protection = var.environment == "prod"
}

resource "google_sql_database" "speakfit" {
  name     = "speakfit"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app" {
  name     = "speakfit_app"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = false
}

# -----------------------------------------------------------------------------
# Redis (Memorystore)
# -----------------------------------------------------------------------------

resource "google_redis_instance" "cache" {
  name           = "speakfit-${var.environment}-redis"
  tier           = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.environment == "prod" ? 4 : 1
  region         = var.region
  
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version = "REDIS_7_0"
}

# -----------------------------------------------------------------------------
# Cloud Storage (Audio files)
# -----------------------------------------------------------------------------

resource "google_storage_bucket" "audio" {
  name     = "speakfit-${var.environment}-audio-${var.project_id}"
  location = var.region
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  cors {
    origin          = ["https://${var.domain}", "https://speakfit.app"]
    method          = ["GET", "HEAD", "PUT", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# -----------------------------------------------------------------------------
# API Server (Cloud Run)
# -----------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name     = "speakfit-${var.environment}-api"
  location = var.region
  
  template {
    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }
    
    containers {
      image = "gcr.io/${var.project_id}/speakfit-api:latest"
      
      resources {
        limits = {
          cpu    = "4"
          memory = "8Gi"
        }
      }
      
      env {
        name  = "DATABASE_URL"
        value = "postgresql://${google_sql_user.app.name}:${random_password.db_password.result}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.speakfit.name}"
      }
      
      env {
        name  = "REDIS_URL"
        value = "redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
      }
      
      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.audio.name
      }
      
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
    }
    
    vpc_access {
      network_interfaces {
        network    = google_compute_network.vpc.name
        subnetwork = google_compute_subnetwork.main.name
      }
      egress = "PRIVATE_RANGES_ONLY"
    }
  }
  
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow unauthenticated access to API
resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# -----------------------------------------------------------------------------
# GPU Worker (Compute Engine with T4)
# -----------------------------------------------------------------------------

resource "google_compute_instance_template" "gpu_worker" {
  name_prefix  = "speakfit-${var.environment}-gpu-worker-"
  machine_type = "n1-standard-4"  # 4 vCPU, 15GB RAM
  region       = var.region
  
  disk {
    source_image = "projects/ml-images/global/images/family/common-gpu"
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
    disk_type    = "pd-ssd"
  }
  
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }
  
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = var.environment != "prod"
  }
  
  network_interface {
    subnetwork = google_compute_subnetwork.main.id
  }
  
  metadata_startup_script = <<-EOF
    #!/bin/bash
    # Install NVIDIA drivers
    /opt/deeplearning/install-driver.sh
    
    # Pull and run worker container
    docker pull gcr.io/${var.project_id}/speakfit-gpu-worker:latest
    docker run -d --gpus all \
      -e REDIS_URL="redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}" \
      -e DATABASE_URL="postgresql://${google_sql_user.app.name}:${random_password.db_password.result}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.speakfit.name}" \
      gcr.io/${var.project_id}/speakfit-gpu-worker:latest
  EOF
  
  service_account {
    email  = google_service_account.gpu_worker.email
    scopes = ["cloud-platform"]
  }
  
  tags = ["gpu-worker"]
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_instance_group_manager" "gpu_workers" {
  name               = "speakfit-${var.environment}-gpu-workers"
  base_instance_name = "gpu-worker"
  zone               = "${var.region}-a"
  target_size        = var.gpu_worker_count
  
  version {
    instance_template = google_compute_instance_template.gpu_worker.id
  }
  
  auto_healing_policies {
    health_check      = google_compute_health_check.gpu_worker.id
    initial_delay_sec = 300
  }
}

resource "google_compute_health_check" "gpu_worker" {
  name               = "speakfit-${var.environment}-gpu-worker-health"
  check_interval_sec = 30
  timeout_sec        = 10
  
  tcp_health_check {
    port = 8080
  }
}

# -----------------------------------------------------------------------------
# Service Accounts
# -----------------------------------------------------------------------------

resource "google_service_account" "gpu_worker" {
  account_id   = "speakfit-${var.environment}-gpu-worker"
  display_name = "SpeakFit GPU Worker"
}

resource "google_project_iam_member" "gpu_worker_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.gpu_worker.email}"
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "database_connection" {
  value     = "postgresql://${google_sql_user.app.name}:***@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.speakfit.name}"
  sensitive = true
}

output "redis_host" {
  value = google_redis_instance.cache.host
}

output "audio_bucket" {
  value = google_storage_bucket.audio.name
}
