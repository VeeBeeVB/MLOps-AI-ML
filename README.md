# MLFLow Dockerize 

**Step1: Directory structure** 
mlops/
├── docker-compose.yml
├── mlflow/
│   └── artifacts/
└── airflow/
    ├── dags/
    ├── logs/


**Step 2: Prepare the docker-compose.yml file**
- Refer the docker compose file. 
- Then run "docker compose up -d"
- Verify using "docker compose ps" or "docker compose ps -a"
- Then check localhost:5000 (mlflow) and localhost:8080 (airflow)

Note: Before run the above command make sure delete other containers with the similar name like "mlflow or modify the container_name in the docker-compose.yml file***"

**Step 3: Initialize the Airflow DB**
docker compose run --rm airflow-webserver airflow db init
docker compose run --rm airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com


docker compose restart airflow-webserver airflow-scheduler

- Now verify http://localhost:8080

**Step 3: Restarting the Containers**
docker compose down
docker compose up -d
docker compose run --rm airflow-webserver airflow db init

(((((((((OR)))))))))

docker compose down -v
docker compose up -d airflow-db mlflow-db mlflow
docker compose run --rm airflow-webserver airflow db migrate
docker compose run --rm airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
docker compose up -d


Note: Make sure you run all these three commands to restrat/stop and start

================================================================================================
**Docker compose yaml file explanation** - Refer the updated code docker-compose.yml file 

version: "3.9"  # Docker Compose file format version (not Docker Engine version)

services:       # All containers (services) are defined under this key

  # ======================
  # MLflow Metadata DB
  # ======================
  mlflow-db:                     # SERVICE NAME (also used as hostname inside Docker network)
    image: postgres:13           # Pull PostgreSQL image version 13 from Docker Hub
    container_name: mlflow-db    # Explicit container name (optional but clear)
    environment:                 # Environment variables passed into the container
      POSTGRES_USER: mlflow      # DB username
      POSTGRES_PASSWORD: mlflow # DB password
      POSTGRES_DB: mlflow       # Database name
    volumes:
      - mlflow-db-data:/var/lib/postgresql/data
        # Named volume → persistent storage for PostgreSQL data

  # ======================
  # MLflow Server
  # ======================
  mlflow:                                  # SERVICE NAME (hostname = mlflow)
    image: ghcr.io/mlflow/mlflow:v2.10.2  # Official MLflow Docker image
    container_name: mlflow                # Explicit container name
    depends_on:
      - mlflow-db                         # Start mlflow-db before mlflow
    ports:
      - "5000:5000"                       # Host:Container port mapping
    environment:
      # SQLAlchemy connection string to MLflow backend DB
      # mlflow-db is resolved via Docker internal DNS
      MLFLOW_BACKEND_STORE_URI: postgresql+psycopg2://mlflow:mlflow@mlflow-db/mlflow

      # Default location where artifacts (models, metrics) are stored
      MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts

    volumes:
      - ./mlflow/artifacts:/mlflow/artifacts
        # Bind mount → host folder mapped into container

    command: >                             # Override default container command
      sh -c "
      pip install --no-cache-dir psycopg2-binary &&   # Install Postgres driver
      mlflow server                                   # Start MLflow server
      --host 0.0.0.0                                  # Listen on all interfaces
      --port 5000                                     # Expose MLflow on port 5000
      "

  # ======================
  # Airflow Metadata DB
  # ======================
  airflow-db:                   # SERVICE NAME (hostname = airflow-db)
    image: postgres:13          # PostgreSQL image
    environment:
      POSTGRES_USER: airflow    # DB username
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow-db-data:/var/lib/postgresql/data
        # Persistent metadata DB for Airflow

  # ======================
  # Airflow Webserver
  # ======================
  airflow-webserver:                   # SERVICE NAME (hostname = airflow-webserver)
    image: apache/airflow:2.8.1         # Official Apache Airflow image
    depends_on:
      - airflow-db                      # Ensure DB starts first
      - mlflow                          # Ensure MLflow is reachable
    ports:
      - "8080:8080"                     # Airflow UI exposed on localhost:8080
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
        # Execution model (industry default for small/medium setups)

      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-db/airflow
        # Airflow metadata DB connection string
        # airflow-db is resolved via Docker DNS

      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
        # Disable example DAGs

      MLFLOW_TRACKING_URI: http://mlflow:5000
        # MLflow service hostname + port
        # "mlflow" comes from the SERVICE NAME above

    volumes:
      - ./airflow/dags:/opt/airflow/dags   # DAG definitions
      - ./airflow/logs:/opt/airflow/logs   # Airflow logs

    command: webserver                    # Start Airflow webserver process

  # ======================
  # Airflow Scheduler
  # ======================
  airflow-scheduler:                     # SERVICE NAME (hostname = airflow-scheduler)
    image: apache/airflow:2.8.1
    depends_on:
      - airflow-webserver                # Start after webserver
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-db/airflow
      MLFLOW_TRACKING_URI: http://mlflow:5000
        # Scheduler also needs MLflow access
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/logs:/opt/airflow/logs
    command: scheduler                   # Start scheduler process

# ======================
# Named Volumes
# ======================
volumes:
  mlflow-db-data:        # Named volume for MLflow Postgres data
  airflow-db-data:       # Named volume for Airflow Postgres data
