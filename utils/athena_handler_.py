import boto3
from botocore.exceptions import ClientError
import time
import streamlit as st
import json

def start_athena_session_(
    workgroup_name,
    description="Starting Athena session",
    coordinator_dpu_size=1,
    max_concurrent_dpus=60,
    default_executor_dpu_size=1,
    additional_configs=None,
    spark_properties=None,
    # notebook_version="Athena notebook version 1",
    session_idle_timeout_in_minutes=None,
    client_request_token=None
):
    """
    Start an Athena session using boto3.

    Args:
    workgroup_name (str): The name of the workgroup.
    description (str): A description of the session. Default is "Starting Athena session".
    coordinator_dpu_size (int): The size of the coordinator DPU. Default is 1.
    max_concurrent_dpus (int): The maximum number of concurrent DPUs. Default is 20.
    default_executor_dpu_size (int): The default size of executor DPUs. Default is 1.
    additional_configs (dict): Additional configurations. Default is None.
    spark_properties (dict): Spark properties. Default is None.
    notebook_version (str): The version of the Athena notebook. Default is "Athena notebook version 1".
    session_idle_timeout_in_minutes (int): The idle timeout for the session in minutes. Default is None.
    client_request_token (str): A unique, case-sensitive identifier that you provide to ensure the idempotency of the request. Default is None.

    Returns:
    dict: A dictionary containing the SessionId and State of the started session.
    """

    # Create an Athena client
    athena_client = boto3.client('athena')

    # Define the engine configuration
    engine_configuration = {
        'CoordinatorDpuSize': coordinator_dpu_size,
        'MaxConcurrentDpus': max_concurrent_dpus,
        'DefaultExecutorDpuSize': default_executor_dpu_size,
    }

    if additional_configs:
        engine_configuration['AdditionalConfigs'] = additional_configs

    if spark_properties:
        engine_configuration['SparkProperties'] = spark_properties

    # Prepare the request parameters
    request_params = {
        'Description': description,
        'WorkGroup': workgroup_name,
        'EngineConfiguration': engine_configuration,
        # 'NotebookVersion': notebook_version
    }

    if session_idle_timeout_in_minutes is not None:
        request_params['SessionIdleTimeoutInMinutes'] = session_idle_timeout_in_minutes

    if client_request_token:
        request_params['ClientRequestToken'] = client_request_token

    try:
        # Start the Athena session
        response = athena_client.start_session(**request_params)

        # Extract relevant information
        session_info = {
            'SessionId': response['SessionId'],
            'State': response['State']
        }

        print(f"Athena session started successfully.")
        print(f"Session ID: {session_info['SessionId']}")
        print(f"State: {session_info['State']}")

        return session_info

    except ClientError as e:
        print(f"An error occurred while starting the Athena session: {e.response['Error']['Message']}")
        return None


class SessionFailedException(Exception):
    """Custom exception for when the session is in a FAILED state."""
    pass

class SessionTimeoutException(Exception):
    """Custom exception for when the session check times out."""
    pass

def wait_for_session_status(session_id, max_wait_seconds=300, check_interval_seconds=3):
    """
    Wait for an Athena session to reach either IDLE or FAILED state.

    Args:
    session_id (str): The ID of the session to check.
    max_wait_seconds (int): Maximum time to wait in seconds. Default is 300 seconds (5 minutes).
    check_interval_seconds (int): Time to wait between status checks in seconds. Default is 10 seconds.

    Returns:
    bool: True if the session state is IDLE.

    Raises:
    SessionFailedException: If the session state is FAILED.
    SessionTimeoutException: If the maximum wait time is exceeded.
    ClientError: If there's an error in the AWS API call.
    """

    athena_client = boto3.client('athena')
    start_time = time.time()

    while True:
        try:
            response = athena_client.get_session_status(SessionId=session_id)
            state = response['Status']['State']

            print(f"Session {session_id} is in state: {state}")

            if state == 'IDLE':
                return True
            elif state == 'FAILED':
                reason = response['Status'].get('StateChangeReason', 'No reason provided')
                raise SessionFailedException(f"Session {session_id} has FAILED. Reason: {reason}")
            elif state == 'TERMINATED':
                # return f"Session {session_id} is in state: {state}"
                return False

            # Check if we've exceeded the maximum wait time
            if time.time() - start_time > max_wait_seconds:
                raise SessionTimeoutException(f"Timeout waiting for session {session_id} to become IDLE or FAILED")

            # Wait for the specified interval before checking again
            time.sleep(check_interval_seconds)

        except ClientError as e:
            print(f"An error occurred while checking the session status: {e.response['Error']['Message']}")
            raise

def execute_athena_calculation(session_id, code_block, workgroup, max_wait_seconds=600, check_interval_seconds=5):
    """
    Execute a calculation in Athena, wait for completion, and retrieve results.

    Args:
    session_id (str): The Athena session ID.
    code_block (str): The code to execute.
    max_wait_seconds (int): Maximum time to wait for execution in seconds. Default is 600 seconds (10 minutes).
    check_interval_seconds (int): Time to wait between status checks in seconds. Default is 10 seconds.

    Returns:
    dict: A dictionary containing execution results or error information.
    """
    athena_client = boto3.client('athena')
    s3_client = boto3.client('s3')

    def start_calculation():
        try:
            response = athena_client.start_calculation_execution(
                SessionId=session_id,
                CodeBlock=code_block,
                # ClientRequestToken=f"token-{time.time()}"  # Unique token for idempotency
            )
            return response['CalculationExecutionId']
        except ClientError as e:
            print(f"Failed to start calculation: {e}")
            raise

    def check_calculation_status(calculation_id):
        try:
            response = athena_client.get_calculation_execution(CalculationExecutionId=calculation_id)
            return response['Status']['State']
        except ClientError as e:
            print(f"Failed to get calculation status: {e}")
            raise

    def get_calculation_result(calculation_id):
        try:
            response = athena_client.get_calculation_execution(CalculationExecutionId=calculation_id)
            return response['Result']
        except ClientError as e:
            print(f"Failed to get calculation result: {e}")
            raise

    def download_s3_file(s3_uri):
        try:
            bucket, key = s3_uri.replace("s3://", "").split("/", 1)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            print(f"Failed to download S3 file: {e}")
            return None

    if session_id:
        if not wait_for_session_status(session_id):
            # Start Session
            session = start_athena_session_(
                workgroup_name=workgroup,
                session_idle_timeout_in_minutes=60
            )
            if wait_for_session_status(session['SessionId']):
                session_id = session['SessionId']
                st.session_state['athena-session'] = session['SessionId']
                
    else:
        session = start_athena_session_(
                workgroup_name=workgroup,
                session_idle_timeout_in_minutes=60
            )
        if wait_for_session_status(session['SessionId']):
                session_id = session['SessionId']
                st.session_state['athena-session'] = session['SessionId']

    # Start the calculation
    calculation_id = start_calculation()
    print(f"Started calculation with ID: {calculation_id}")

    # Wait for the calculation to complete
    start_time = time.time()
    while True:
        status = check_calculation_status(calculation_id)
        print(f"Calculation status: {status}")

        if status in ['COMPLETED', 'FAILED', 'CANCELED']:
            break

        if time.time() - start_time > max_wait_seconds:
            print("Calculation timed out")
            return {"error": "Calculation timed out"}

        time.sleep(check_interval_seconds)

    # Get the calculation result
    result = get_calculation_result(calculation_id)

    if status == 'COMPLETED':
        # Download and return the result
        result_content = download_s3_file(result['ResultS3Uri'])
        print(result)
        return {
            "status": "COMPLETED",
            "result": result_content,
            "stdout": download_s3_file(result['StdOutS3Uri']),
            "stderr": download_s3_file(result['StdErrorS3Uri'])
        }
    elif status == 'FAILED':
        # Get the error file
        error_content = download_s3_file(result['StdErrorS3Uri'])
        return {
            "status": "FAILED",
            "error": error_content,
            "stdout": download_s3_file(result['StdOutS3Uri'])
        }
    else:
        return {"status": status}

def send_athena_job(payload, workgroup):
    """
    Send a Spark job payload to the specified URL.

    Args:
    payload (dict): The payload containing the Spark job details.

    Returns:
    dict: The response from the server.
    """    

    code_block = """
import boto3
import os
def put_obj_in_s3_bucket_(docs, bucket, key_prefix):
    S3 = boto3.client('s3')
    if isinstance(docs,str):
        file_name=os.path.basename(docs)
        file_path=f"{key_prefix}/{docs}"
        S3.upload_file(f"/tmp/{docs}", bucket, file_path)
    else:
        file_name=os.path.basename(docs.name)
        file_path=f"{key_prefix}/{file_name}"
        S3.put_object(Body=docs.read(),Bucket= BUCKET, Key=file_path)           
    return f"s3://{bucket}/{file_path}"

def handle_results_(result, bucket, s3_file_path):
    image_holder = []    
    if isinstance(result, dict):
        for item, value in result.items():
            if "plotly-files" in item and value is not None:
                if isinstance(value, list):
                    for img in value:
                        image_path_s3 = put_obj_in_s3_bucket_(img, bucket, s3_file_path)
                        image_holder.append(image_path_s3)                            
                else:                        
                    image_path_s3 = put_obj_in_s3_bucket_(value,bucket,s3_file_path)
                    image_holder.append(image_path_s3)
    
    tool_result = {
        "result": result,            
        "plotly": image_holder
    }
    return tool_result

# iterate = input_data.get('iterate', 0)
bucket="BUCKET-NAME"
s3_file_path="BUCKET-PATH"
result_final = handle_results_(output, bucket, s3_file_path)
print(f"<output>{result_final}</output>")
""".replace("BUCKET-NAME",payload["bucket"]).replace("BUCKET-PATH", payload["file_path"])

    try:
        result = execute_athena_calculation(st.session_state['athena-session'], payload["code"]+ "\n" +code_block, workgroup)
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return e