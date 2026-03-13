# How to run Spark container with Podman

## Start spark container with mounted volume and working directory set to /workspace (where our data and code are)

```bash
podman run -d --name spark \
  --user 0 \
  -v "$(pwd):/workspace:Z" \
  -w /workspace \
  apache/spark-py \
  tail -f /dev/null
  ```

  inside container run

- to get into workspace fodler

    ```bash
    cd workspace
    ``

- to start streaming

    ```bash
    spark-submit spark_job.py 
    ```

## Find schreenshots with results under /screenshots folder