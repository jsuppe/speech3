#!/bin/bash
# Setup PostgreSQL for SpeakFit development
# Run with: sudo bash setup_postgres.sh

set -e

echo "=== SpeakFit PostgreSQL Setup ==="
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash $0"
    exit 1
fi

# Install PostgreSQL
echo "Installing PostgreSQL..."
apt-get update
apt-get install -y postgresql postgresql-contrib

# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Create user and database
echo "Creating database and user..."
sudo -u postgres psql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'speakfit') THEN
        CREATE ROLE speakfit WITH LOGIN PASSWORD 'speakfit_dev_2024';
    END IF;
END
\$\$;

-- Create database
DROP DATABASE IF EXISTS speakfit;
CREATE DATABASE speakfit OWNER speakfit;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE speakfit TO speakfit;
EOF

echo
echo "=== PostgreSQL Setup Complete ==="
echo
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: speakfit"
echo "  User: speakfit"
echo "  Password: speakfit_dev_2024"
echo
echo "Connection URL:"
echo "  postgresql://speakfit:speakfit_dev_2024@localhost:5432/speakfit"
echo
echo "To migrate USER DATA (cloud-ready), run:"
echo "  cd /home/melchior/speech3"
echo "  source venv/bin/activate"
echo "  pip install psycopg2-binary"
echo "  python scripts/migrate_user_data.py --postgres-url 'postgresql://speakfit:speakfit_dev_2024@localhost:5432/speakfit'"
echo
echo "This migrates ONLY:"
echo "  - 5 users"
echo "  - ~150 user recordings (14MB audio)"
echo "  - Analyses, transcriptions, coach messages"
echo
echo "Batch imports (114K) stay in local SQLite for training/benchmarks."
echo
