from databricks.sdk.runtime import dbutils, spark


def get_databricks_token():
    """ Memorable wrapper to get Databricks token """
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def set_spark_env_variable(key: str, val: str):
    """ Wrapper to set Spark environment variable """
    # Add default values for env key, val to use this function
    spark.conf.set(key, val)

def get_llm_endpoint_base_url():
    """ Wrapper to get LLM endpoint base url """
    return # Add env-specific endpoint URL here to use this function
